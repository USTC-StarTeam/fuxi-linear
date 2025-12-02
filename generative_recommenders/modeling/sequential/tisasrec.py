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

import numpy as np

import sys
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

FLOAT_MIN = -sys.float_info.max

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, ffn_hidden_units): # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, ffn_hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(ffn_hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(queries.device)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs

class TiSASRec(GeneralizedInteractionModule):
    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        time_span: int,
        dropout_rate: float,
        ffn_hidden_dim: int,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        activation_checkpoint: bool = False,
        verbose: bool = False,
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

        self.abs_pos_K_emb = torch.nn.Embedding(self._max_sequence_length, embedding_dim)
        self.abs_pos_V_emb = torch.nn.Embedding(self._max_sequence_length, embedding_dim)
        self.time_matrix_K_emb = torch.nn.Embedding(time_span+1, embedding_dim)
        self.time_matrix_V_emb = torch.nn.Embedding(time_span+1, embedding_dim)

        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=dropout_rate)
        
        # self.last_layernorm = torch.nn.LayerNorm(embedding_dim, eps=1e-8)

        self._ffn_hidden_dim: int = ffn_hidden_dim

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        self.time_span = time_span
        
        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(embedding_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(
                embedding_dim,
                num_heads,
                dropout_rate,
                None,
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(embedding_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                embedding_dim, 
                dropout_rate,
                ffn_hidden_dim,
            )
            self.forward_layers.append(new_fwd_layer)
        
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

    def reset_state(self) -> None:
        for name, params in self.named_parameters():
            if (
                "_input_features_preproc" in name
                or "_embedding_module" in name
                or "_output_postproc" in name
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
            f"TiSASRec-d{self._item_embedding_dim}-b{self._num_blocks}-h{self._num_heads}"
            + "-"
            + self._input_features_preproc.debug_str()
            + "-"
            + self._output_postproc.debug_str()
            + f"-ffn{self._ffn_hidden_dim}"
            + f"{'-ac' if self._activation_checkpoint else ''}"
        )

    def _run_one_layer(
        self,
        i: int,
        user_embeddings: torch.Tensor,
        timeline_mask: torch.Tensor,
        invalid_attn_mask,
        time_matrix_K, 
        time_matrix_V,
        abs_pos_K, 
        abs_pos_V,
    ) -> torch.Tensor:
        Q = self.attention_layernorms[i](user_embeddings) # PyTorch mha requires time first fmt
        mha_outputs = self.attention_layers[i](
            Q, user_embeddings,
            timeline_mask, invalid_attn_mask,
            time_matrix_K, time_matrix_V,
            abs_pos_K, abs_pos_V
        )
        user_embeddings = Q + mha_outputs
            
        user_embeddings = self.forward_layernorms[i](user_embeddings)
        user_embeddings = self.forward_layers[i](user_embeddings)
        user_embeddings *= ~timeline_mask.unsqueeze(-1)
        return user_embeddings

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
        
        time_seq = past_payloads['timestamps']
        
        self.dev = user_embeddings.device
        B, n, d = user_embeddings.shape
        
        diff_matrix = torch.abs(time_seq.unsqueeze(2) - time_seq.unsqueeze(1))
        min_vals, _ = torch.min(torch.clamp(diff_matrix, min=1).reshape(B, -1), dim=1, keepdim=True)
        time_matrices = torch.clamp((torch.floor(diff_matrix / min_vals.unsqueeze(-1)) + 0.1).to(torch.long), max=self.time_span)
        
        positions = np.tile(np.array(range(n)), [B, 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)
        
        # time_matrices =time
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        timeline_mask = ~valid_mask.to(torch.bool).squeeze(-1)
        invalid_attn_mask = self._attn_mask

        for i in range(len(self.attention_layers)):
            if self._activation_checkpoint:
                user_embeddings = torch.utils.checkpoint.checkpoint(
                    self._run_one_layer,
                    i,
                    user_embeddings, 
                    timeline_mask,
                    invalid_attn_mask,
                    time_matrix_K, 
                    time_matrix_V,
                    abs_pos_K, 
                    abs_pos_V,
                    use_reentrant=False,
                )
            else:
                user_embeddings = self._run_one_layer(i, user_embeddings, timeline_mask,
                    invalid_attn_mask,
                    time_matrix_K, 
                    time_matrix_V,
                    abs_pos_K, 
                    abs_pos_V,)

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
