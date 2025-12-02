# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

from typing import Dict, Optional, Tuple

import torch
from torch import nn

import torch.nn.functional as F

from generative_recommenders.modeling.ndp_module import NDPModule
from generative_recommenders.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.modeling.sequential.utils import get_current_embeddings
from generative_recommenders.modeling.similarity_module import (
    GeneralizedInteractionModule,
)

from .ttt import TTTConfig, TTTModel

class TTT4Rec(GeneralizedInteractionModule):

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        activation_checkpoint: bool = False,
        verbose: bool = False,
        intermediate_size=256,
        rope_theta=10000.0,
        ttt_layer_type="mlp",
        ttt_base_lr=1.0,
        mini_batch_size=8,
        use_gate=False,
        pre_conv=False,
        share_qk=False,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_module: EmbeddingModule = embedding_module
        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len + max_output_len
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._activation_checkpoint: bool = activation_checkpoint
        self._verbose: bool = verbose

        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads

        ttt_configs = TTTConfig(
          hidden_size=embedding_dim,
          intermediate_size=intermediate_size,
          num_hidden_layers=num_blocks,
          num_attention_heads=num_heads,
          max_position_embeddings=self._max_sequence_length,
          rope_theta=rope_theta,
          ttt_layer_type=ttt_layer_type,
          ttt_base_lr=ttt_base_lr,
          mini_batch_size=mini_batch_size,
          use_gate=use_gate,
          pre_conv=pre_conv,
          share_qk=share_qk,
        )
        
        self._ttt_layers = TTTModel(ttt_configs)
        del self._ttt_layers.embed_tokens
        
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (self._max_sequence_length, self._max_sequence_length),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self.reset_state()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reset_state(self) -> None:
        for name, params in self.named_parameters():
            if (
                "_input_features_preproc" in name
                or "_embedding_module" in name
                or "_output_postproc" in name
                or "_ttt_layers" in name
            ):
                if self._verbose:
                    print(f"Skipping initialization for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(
                        f"Initialize {name} as xavier normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")
        self._ttt_layers.apply(self._init_weights)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        return (
            f"TTT4Rec-d{self._item_embedding_dim}-b{self._num_blocks}-h{self._num_heads}"
            + "-"
            + self._input_features_preproc.debug_str()
            + "-"
            + self._output_postproc.debug_str()
            + f"{'-ac' if self._activation_checkpoint else ''}"
        )

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            past_ids: (B, N,) x int

        Returns:
            (B, N, D,) x float
        """
        past_lengths, user_embeddings, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        user_embeddings = self._ttt_layers(
            inputs_embeds=user_embeddings,
            attention_mask=(1-self._attn_mask.long())
        ).last_hidden_state

        return self._output_postproc(user_embeddings)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            past_ids: [B, N] x int64 where the latest engaged ids come first. In
                particular, [:, 0] should correspond to the last engaged values.
            past_ratings: [B, N] x int64.
            past_timestamps: [B, N] x int64.

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings = self.generate_user_embeddings(
            past_lengths,
            past_ids,
            past_embeddings,
            past_payloads,
        )
        return encoded_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,  # [B, N] x int64
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        encoded_seq_embeddings = self.generate_user_embeddings(
            past_lengths, past_ids, past_embeddings, past_payloads
        )  # [B, N, D]
        return get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )

    def predict(
        self,
        past_ids: torch.Tensor,
        past_ratings: torch.Tensor,
        past_timestamps: torch.Tensor,
        next_timestamps: torch.Tensor,
        target_ids: torch.Tensor,
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        return self.interaction(
            self.encode(past_ids, past_ratings, past_timestamps, next_timestamps),  # pyre-ignore [6]
            target_ids,
        )  # [B, X]
