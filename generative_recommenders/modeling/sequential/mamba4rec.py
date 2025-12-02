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

import logging

import torch
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

from mamba_ssm import Mamba

import math

from torch import nn
import numpy as np

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, expand_ffn):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*expand_ffn, dropout=dropout)
    
    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states

class Mamba4Rec(GeneralizedInteractionModule):
    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        expand_ffn: int,
        num_blocks: int,
        dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        use_time_interval_embedding: bool = True,
        activation_checkpoint: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        logging.info(f'use_time_interval_embedding = {use_time_interval_embedding}')
        
        self._use_time_interval = use_time_interval_embedding        
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

        self.hidden_size = embedding_dim
        self.dropout_prob = dropout_rate
        
        # Hyperparameters for Mamba block
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.expand_ffn = expand_ffn
        
        self._use_time_interval = use_time_interval_embedding        
        if use_time_interval_embedding :
            self._emb_t = BucketedTimeIntervalEmbeddingModule(128, embedding_dim)
        
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self._num_blocks,
                expand_ffn=self.expand_ffn,
            ) for _ in range(self._num_blocks)
        ])
        
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
        self.mamba_layers.apply(self._init_weights)
        for name, params in self.named_parameters():
            if (
                "_input_features_preproc" in name
                or "_embedding_module" in name
                or "_output_postproc" in name
                or 'mamba_layers' in name
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

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        return (
            f"Mamba4rec-d{self.hidden_size}-b{self._num_blocks}"
            + "-"
            + self._input_features_preproc.debug_str()
            + "-"
            + self._output_postproc.debug_str()
            + f"-ffn{self.expand_ffn}d"
            + f"{'-ac' if self._activation_checkpoint else ''}"
        )

    def _run_one_layer(
        self,
        i: int,
        user_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self.mamba_layers[i](user_embeddings)
        return user_embeddings * valid_mask

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
        
        if self._use_time_interval :
            user_embeddings = self._emb_t(user_embeddings, past_payloads['timestamps'], valid_mask)

        for i in range(len(self.mamba_layers)):
            if self._activation_checkpoint:
                user_embeddings = torch.utils.checkpoint.checkpoint(
                    self._run_one_layer,
                    i,
                    user_embeddings,
                    valid_mask,
                    use_reentrant=False,
                )
            else:
                user_embeddings = self._run_one_layer(i, user_embeddings, valid_mask)

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
