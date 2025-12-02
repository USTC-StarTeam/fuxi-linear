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

"""
Implements LRURec (Linear Recurrent Units for Sequential Recommendation, https://arxiv.org/abs/2310.02367, WSDM 2024).
"""

from typing import Dict, Optional, Tuple

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

import math

from torch import nn
import numpy as np

class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above 
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)

class LRUBlock(nn.Module):
    def __init__(
        self, 
        embedding_dim,
        ffn_multiply,
        dropout,
        ffn_dropout_rate,
    ):
        super().__init__()
        hidden_size = embedding_dim
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=int(hidden_size*ffn_multiply), dropout=ffn_dropout_rate)
    
    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x

class LRURec(GeneralizedInteractionModule):
    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        ffn_multiply: int,
        dropout_rate: float,
        ffn_dropout_rate: float,
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

        self._lru_blocks = torch.nn.ModuleList()
        self._num_blocks: int = num_blocks
        ffn_hidden_dim = int(embedding_dim * embedding_dim)
        self._ffn_hidden_dim: int = ffn_hidden_dim
        self._ffn_dropout_rate: float = ffn_dropout_rate

        for _ in range(num_blocks):
            self._lru_blocks.append(LRUBlock(
                embedding_dim=embedding_dim,
                ffn_multiply=ffn_multiply,
                dropout=dropout_rate,
                ffn_dropout_rate=ffn_dropout_rate,
            ))

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

    def reset_state(self, mean=0, std=0.02, lower=-0.04, upper=0.04) -> None:
        with torch.no_grad() :
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
            
            for name, params in self.named_parameters():
                if (
                    "_input_features_preproc" in name
                    or "_embedding_module" in name
                    or "_output_postproc" in name
                ):
                    if self._verbose:
                        print(f"Skipping initialization for {name}")
                    continue
                n = name
                p = params
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        return (
            f"LRURec-d{self._item_embedding_dim}-b{self._num_blocks}"
            + "-"
            + self._input_features_preproc.debug_str()
            + "-"
            + self._output_postproc.debug_str()
            + f"-ffn{self._ffn_hidden_dim}-d{self._ffn_dropout_rate}"
            + f"{'-ac' if self._activation_checkpoint else ''}"
        )

    def _run_one_layer(
        self,
        i: int,
        user_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self._lru_blocks[i](user_embeddings, valid_mask)
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

        B, N, D = user_embeddings.shape
        log2_L = int(np.ceil(np.log2(N)))
        N2 = 2 ** log2_L
        user_embeddings = torch.concat([
                user_embeddings,
                torch.zeros(B, N2 - N, D, dtype=user_embeddings.dtype, device=user_embeddings.device)
            ],
            dim=1,
        )
        valid_mask = torch.concat([
                valid_mask,
                torch.zeros_like(valid_mask[:, :N2-N])
            ],
            dim=1,
        )
        
        for i in range(len(self._lru_blocks)):
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

        return self._output_postproc(user_embeddings[:, :N])

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
