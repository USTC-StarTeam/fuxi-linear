import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from generative_recommenders.modeling.ndp_module import NDPModule

import einops
import logging

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

from generative_recommenders.modeling.sequential.hstu import RelativeAttentionBiasModule

TIMESTAMPS_KEY = "timestamps"

class SeperatedRelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias, rel_ts_bias

class FunctionalRelativeAttentionBias (RelativeAttentionBiasModule) :
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """
    def __init__(
        self,
        max_seq_len: int,
        func_type: Literal['linear', 'log', 'exp', 'sin', 'pow', 'mixed', 'nn', 'zero'],
    ) -> None:
        super().__init__()
        self._max_seq_len = max_seq_len
        
        self._func_type = func_type
        
        if func_type == 'linear' :
            # a * x + b
            self._lin_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.01, 0.01))
            self._lin_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
        elif func_type == 'log' :
            # a * log(1 + b * x) + c
            self._log_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.01, 0.01))
            self._log_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.5, 1))
            self._log_c = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.05, 0.05))
        elif func_type == 'exp' :
            # a * exp(-b * x)
            self._exp_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._exp_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-2, 0))
        elif func_type == 'sin' :
            # c * sin(a * x + b) + d
            self._sin_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.02, 0.02))
            self._sin_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-torch.pi, torch.pi))
            self._sin_c = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-2, 2))
            self._sin_d = nn.Parameter(torch.empty(1, dtype=torch.float32).zero_())
        elif func_type == 'pow' :
            # a * pow(x, b)
            self._pow_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._pow_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.4, 0.8))
        elif func_type == 'mixed' :
            self._lin_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.01, 0.01))
            self._lin_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._log_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.01, 0.01))
            self._log_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.5, 1))
            self._log_c = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.05, 0.05))
            self._exp_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._exp_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-2, 0))
            self._sin_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.02, 0.02))
            self._sin_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-torch.pi, torch.pi))
            self._sin_c = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-2, 2))
            self._sin_d = nn.Parameter(torch.empty(1, dtype=torch.float32).zero_())
            self._pow_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._pow_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.4, 0.8))
        elif func_type == 'nn' :
            self._nn_a = nn.Linear(1, 5)
            self._nn_b = nn.Linear(5, 5)
            self._nn_c = nn.Linear(5, 1)
        elif func_type == 'zero' :
            pass
        else :
            raise Exception(f'Unknown function type {func_type}')
        
        self._func_map = {
            'linear': self.f_lin,
            'log': self.f_log,
            'exp': self.f_exp,
            'sin': self.f_sin,
            'pow': self.f_pow,
            'mixed': self.f_mix,
            'nn': self.f_nn,
            'zero': self.f_zero,
        }
        
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
    
    def f_lin(self, x) :
        return self._lin_a * x + self._lin_b
    
    def f_log(self, x) :
        x = torch.relu(x)
        assert self._log_b > 0
        return self._log_a * torch.log(1 + self._log_b * x) + self._log_c
    
    def f_exp(self, x) :
        x = torch.relu(x)
        exp_b = torch.exp(self._exp_b)
        assert exp_b > 0
        return self._exp_a * torch.exp(-exp_b * x)
    
    def f_sin(self, x) :
        return self._sin_c * torch.sin(self._sin_a * x + self._sin_b) + self._sin_d
    
    def f_pow(self, x) :
        x = torch.relu(x) + 1
        return self._pow_a * torch.pow(x, -self._pow_b)
    
    def f_mix(self, x) :
        return (self.f_lin(x) + self.f_log(x) + self.f_exp(x) + self.f_sin(x) + self.f_pow(x)) / 5    
    
    def f_nn(self, x) :
        x = self._nn_a(x.to(torch.float32).unsqueeze(-1))
        x = self._nn_b(torch.sin(x))
        x = F.silu(x)
        return self._nn_c(x).squeeze(-1)
    
    def f_zero(self, x) :
        return torch.zeros_like(x, device=x.device)
        
    def f(self, x) :
        func = self._func_map[self._func_type]
        return func(x)
    
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        ext_timestamps = ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
        
        rel_ts_bias = self.f(ext_timestamps)
        
        rel_pos_bias = t[:, :, r:-r]
        return rel_pos_bias, rel_ts_bias
    

class MultistageFeedforwardNeuralNetwork(torch.nn.Module) :
    def __init__(
        self, 
        ams_output_size, 
        input_size, 
        hidden_size, 
        output_size, 
        dropout_ratio: float,
        bias: bool = False, 
        single_stage: bool = False,
        epsilon: float = 1e-6,
    ) :
        super(MultistageFeedforwardNeuralNetwork, self).__init__()
        self.lin0 = torch.nn.Linear(ams_output_size, input_size)
        self.is_single_stage = single_stage
        self.dropout_ratio = dropout_ratio
        self.input_size = input_size
        self.eps = epsilon
        if not single_stage :
            self.lin1 = torch.nn.Linear(input_size, hidden_size, bias=bias)
            self.lin2 = torch.nn.Linear(hidden_size, output_size, bias=bias)
            self.lin3 = torch.nn.Linear(input_size, hidden_size, bias=bias)
    
    def forward(self, X, X0) :
        X = (
            self.lin0(
                F.dropout(
                    X,
                    p = self.dropout_ratio,
                    training = self.training
                )
            ) + X0
        )
        if not self.is_single_stage :
            normed_X = F.rms_norm(X, normalized_shape=[self.input_size], eps=self.eps)
            normed_X = F.dropout(
                normed_X,
                p = self.dropout_ratio,
                training = self.training
            )
            X1 = F.silu(self.lin1(normed_X)) * self.lin3(normed_X)
            X = self.lin2(X1) + X
        return X
    
    def init(self) :
        torch.nn.init.xavier_uniform_(self.lin0.weight)
        if not self.is_single_stage :
            torch.nn.init.xavier_uniform_(self.lin1.weight)
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            torch.nn.init.xavier_uniform_(self.lin3.weight)
        
class FuXiBetaBlockJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        epsilon: float = 1e-6,
        ffn_multiply: float = 1,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    embedding_dim,
                    linear_hidden_dim * num_heads * 3,
                )
            ).normal_(mean=0, std=0.02),
        )
        self._linear_activation: str = linear_activation
        
        self._mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size = linear_hidden_dim * num_heads * 2,
            input_size = embedding_dim,
            hidden_size = int(embedding_dim * ffn_multiply),
            output_size = embedding_dim,
            dropout_ratio = dropout_ratio,
            single_stage = False,
            epsilon = epsilon
        )
        self._mffn.init()
        
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads * 2], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
    ):
        n: int = invalid_attn_mask.size(-1)
            
        normed_x = self._norm_input(x)

        batched_mm_output = torch.mm(normed_x, self._uvqk)
        if self._linear_activation == "silu":
            batched_mm_output = F.silu(batched_mm_output)
        elif self._linear_activation == "none":
            batched_mm_output = batched_mm_output
        u, v = torch.split(
            batched_mm_output,
            [
                self._linear_dim * self._num_heads * 2,
                self._linear_dim * self._num_heads,
            ],
            dim=1,
        )

        B: int = x_offsets.size(0) - 1
        
        pos_attn, ts_attn = self._rel_attn_bias(all_timestamps)
        pos_attn = pos_attn * invalid_attn_mask.unsqueeze(0)
        ts_attn = ts_attn * invalid_attn_mask.unsqueeze(0)
                
        padded_v = torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
            B, n, self._num_heads, self._linear_dim
        )
                
        output_pos = torch.einsum(
            "bnm,bmhd->bnhd",
            pos_attn,
            padded_v,
        )
        output_ts = torch.einsum(
            "bnm,bmhd->bnhd",
            ts_attn,
            padded_v,
        )
                
        combined_output = torch.concat(
            [output_pos, output_ts],
            dim=-1
        ).reshape(B, n, self._num_heads * self._linear_dim * 2)
                
        attn_output = torch.ops.fbgemm.dense_to_jagged(
            combined_output, [x_offsets],
        )[0]

        attn_output = u * self._norm_attn_output(attn_output)

        new_outputs = self._mffn(attn_output, x)

        return new_outputs


class FuXiBetaJagged(torch.nn.Module):

    def __init__(
        self,
        modules: List[FuXiBetaBlockJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
    ) -> torch.Tensor :

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                x = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                )

        return x

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        if len(x.size()) == 3:
            x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]

        jagged_x = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
        )
        y = torch.ops.fbgemm.jagged_to_padded_dense(
            values=jagged_x,
            offsets=[x_offsets],
            max_lengths=[invalid_attn_mask.size(1)],
            padding_value=0.0,
        )
        return y


class FuXiBeta(GeneralizedInteractionModule):
    """
    Implements FuXi Block
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        ffn_multiply: int,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        verbose: bool = True,
        func_type: Literal['linear', 'exp', 'log', 'pow', 'mixed', 'sin', 'nn'] = 'pow'
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        self._fuxi = FuXiBetaJagged(
            modules=[
                FuXiBetaBlockJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    relative_attention_bias_module=(
                        FunctionalRelativeAttentionBias(
                            max_seq_len=max_output_len + max_sequence_len,
                            func_type=func_type
                        )
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    ffn_multiply=ffn_multiply,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if ("_fuxi" in name) or ("_embedding_module" in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
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
        debug_str = (
            f"FuXi-alpha-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor :
        """
        [B, N] -> [B, N, D].
        """
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        float_dtype = user_embeddings.dtype
        user_embeddings = self._fuxi(
            x=user_embeddings,
            x_offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths),
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
        )
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
        Runs the main encoder.

        Args:
            past_lengths: (B,) x int64
            past_ids: (B, N,) x int64 where the latest engaged ids come first. In
                particular, past_ids[i, past_lengths[i] - 1] should correspond to
                the latest engaged values.
            past_embeddings: (B, N, D) x float or (\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            past_lengths: (B,) x int64.
            past_ids: (B, N,) x int64.
            past_embeddings: (B, N, D,) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            return_cache_states: bool.

        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        encoded_seq_embeddings = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )  # [B, N, D]
        current_embeddings = get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )
        return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Runs encoder to obtain the current hidden states.

        Args:
            past_lengths: (B,) x int.
            past_ids: (B, N,) x int.
            past_embeddings: (B, N, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            (B, D,) x float, representing encoded states at the most recent time step.
        """
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
