# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/6/13

import math
import torch
from einops import repeat
from torch import nn
from .ssd import TiSSD

from typing import Callable, Dict, List, Optional, Tuple, Union

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

import torch.distributed as dist

class TiM4Rec(GeneralizedInteractionModule):

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        
        # Tim4Rec
        num_blocks: int,
        dropout_prob: float,
        time_drop_out: float,
        
        # SSDLayer
        d_state: int,
        d_conv: int,
        expand: int,
        head_dim: int,
        chunk_size: int,
        is_ffn,
        is_time, 
        p2p_residual,
        norm_eps: float,
        is_kai_ming_init: bool,
        
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

        # Hyperparameters for TiM4Rec
        self.hidden_size = embedding_dim
        self.num_layers = num_blocks
        self.dropout_prob = dropout_prob
        self.time_drop_out = time_drop_out
        
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.is_ffn = is_ffn
        self.is_time = is_time
        self.p2p_residual = p2p_residual
        self.norm_eps = norm_eps
        self.is_kai_ming_init = is_kai_ming_init
        
        assert (self.hidden_size * self.expand) % self.head_dim == 0, \
            f'hidden_size * expand {self.hidden_size * self.expand} can\'t divisible by head_dim {self.head_dim} !'
        self.n_heads = (self.hidden_size * self.expand) // self.head_dim
        
        self._num_blocks: int = num_blocks
        self._num_heads: int = self.n_heads

        if self.is_time:
            # self.time_start_token = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.layer_norm_time = nn.LayerNorm(self._max_sequence_length, eps=self.norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.ssd_layers = nn.ModuleList([
            TiSSDLayer(
                d_model=self.hidden_size,
                seq_len=self._max_sequence_length,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                num_layers=self.num_layers,
                head_dim=self.head_dim,
                chunk_size=self.chunk_size,
                dropout=self.dropout_prob,
                time_drop_out=self.time_drop_out,
                is_ffn=self.is_ffn,
                is_time=self.is_time,
                p2p_residual=self.p2p_residual,
                norm_eps=self.norm_eps
            ) for _ in range(self.num_layers)
        ])
        
        self.reset_state()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            if self.is_kai_ming_init:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
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
                or 'ssd_layers' in name
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
        
        self.ssd_layers.apply(self._init_weights)
    
    def calculate_time_diff(self, time_stamp):
        """
        Calculate the interaction time difference
        :param time_stamp: [batch_size, seq_len]
        :return: [batch_size, seq_len]
        """
        batch_size = time_stamp.shape[0]
        # [batch_size, seq_len - 1]
        time_diff = time_stamp[:, 1:] - time_stamp[:, :-1]
        # add first time diff
        # time_diff = torch.concat([repeat(self.time_start_token, '1 -> b 1', b=batch_size), time_diff], dim=1)
        time_diff = torch.concat([torch.zeros(batch_size, 1).to(time_diff.device), time_diff], dim=1)
        # time_diff = nn.functional.normalize(time_diff, p=2, dim=-1)
        time_diff = self.layer_norm_time(self.dropout(time_diff))
        # [batch_size, seq_len] -> [batch_size, n_heads, seq_len]
        time_diff = repeat(time_diff, 'b l -> b h l', h=self.n_heads)
        return time_diff
    
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        return (
            f"TiM4Rec-d{self._item_embedding_dim}-b{self._num_blocks}-h{self._num_heads}"
            + "-"
            + self._input_features_preproc.debug_str()
            + "-"
            + self._output_postproc.debug_str()
            + f"{'-ac' if self._activation_checkpoint else ''}"
        )

    def _run_one_layer(
        self,
        i: int,
        user_embeddings: torch.Tensor,
        time_diff: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings, time_diff = self.ssd_layers[i](user_embeddings, time_diff)
        user_embeddings *= valid_mask
        return user_embeddings, time_diff

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

        rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())

        timestamps = past_payloads['timestamps']
        time_diff = self.calculate_time_diff(timestamps)
        
        for i in range(len(self.ssd_layers)):
            if self._activation_checkpoint:
                user_embeddings, time_diff = torch.utils.checkpoint.checkpoint(
                    self._run_one_layer,
                    i,
                    user_embeddings,
                    time_diff,
                    valid_mask,
                    use_reentrant=False,
                )
            else:
                user_embeddings, time_diff = self._run_one_layer(i, user_embeddings, time_diff, valid_mask)

        if self.training :
            user_embeddings = user_embeddings + time_diff[:, 0, :].unsqueeze(-1) * 0

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


class TiSSDLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 num_layers: int,
                 head_dim: int,
                 chunk_size: int,
                 dropout: float,
                 time_drop_out: float,
                 is_ffn: bool = True,
                 is_time: bool = True,
                 p2p_residual: bool = False,
                 norm_eps: float = 1e-12):
        """
        A single-layer TiSSDLayer, containing a TiSSDBlock and an FFN(if is_ffn is True)

        :param d_model: vector embedding dimension
        :param d_state: the B, C matrix dimension in SSD
        :param d_conv: causal-conv1d kernel size
        :param expand: coefficient of expanding
        :param num_layers: the number of SSDLayer layers,
                used to determined whether the SSDLayer needs residuals connections
        :param head_dim: Header dimension of an SSD
        :param chunk_size: Chunk size of an SSD
        :param dropout: dropout_radio
        :param time_drop_out: time_dropout_radio
        :param is_ffn: whether the FFN is included
        :param is_time: whether the Time-aware is included
        :param p2p_residual: whether you use point-to-point residuals
        :param norm_eps: normalization epsilon
        """
        super(TiSSDLayer, self).__init__()
        self.num_layers = num_layers
        self.ssd = TiSSD(
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            chunk_size=chunk_size,
            bias=True,
            rms_norm=True,
            time_drop_out=time_drop_out,
            is_time=is_time,
            p2p_residual=p2p_residual,
            norm_eps=norm_eps
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.is_ffn = is_ffn
        if is_ffn:
            self.ffn = FeedForward(
                d_model=d_model,
                inner_size=d_model * 4,
                dropout=dropout
            )

    def forward(self, x, time_diff):
        """
        x -> ssd(x)
        -> ffn(x) if is_ffn is True
        :param x: shape: [batch_size, seq_len, d_model]
        :param time_diff: shape: [batch_size, seq_len]
        :return: shape: [batch_size, seq_len, d_model]
        """
        # hidden = self.layer_norm(x)
        hidden, time_diff = self.ssd(x, time_diff)
        # Determine whether SSDBlock needs residual by num_layers
        if self.num_layers == 1:
            hidden = self.layer_norm(self.dropout(hidden))
        else:
            hidden = self.layer_norm(self.dropout(hidden) + x)

        if self.is_ffn:
            return self.ffn(hidden), time_diff
        else:
            return hidden, time_diff


def gelu(x):
    """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2, norm_eps=1e-12):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.fc2 = nn.Linear(inner_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Hardswish()
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x):
        hidden = self.act(self.fc1(x))
        hidden = self.dropout(hidden)

        hidden = self.fc2(hidden)
        hidden = self.layer_norm(self.dropout(hidden) + x)
        return hidden


if __name__ == '__main__':
    pass
