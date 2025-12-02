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

from typing import Optional

import gin
from generative_recommenders.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.modeling.sequential.hstu import HSTU
from generative_recommenders.modeling.sequential.fuxi_alpha import FuXi
from generative_recommenders.modeling.sequential.fuxi_linear import FuXiLinear
from generative_recommenders.modeling.sequential.fuxi_beta import FuXiBeta
from generative_recommenders.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.modeling.sequential.sasrec import SASRec
from generative_recommenders.modeling.sequential.tisasrec import TiSASRec
from generative_recommenders.modeling.sequential.lru import LRURec
from generative_recommenders.modeling.sequential.mamba4rec import Mamba4Rec
from generative_recommenders.modeling.sequential.tim4rec import TiM4Rec
from generative_recommenders.modeling.sequential.ttt4rec import TTT4Rec

from generative_recommenders.modeling.similarity_module import (
    GeneralizedInteractionModule,
    InteractionModule,
)

@gin.configurable
def sasrec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    ffn_hidden_dim: int = 64,
    ffn_activation_fn: str = "relu",
    ffn_dropout_rate: float = 0.2,
    num_blocks: int = 2,
    num_heads: int = 1,
) -> GeneralizedInteractionModule:
    return SASRec(
        embedding_module=embedding_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_activation_fn=ffn_activation_fn,
        ffn_dropout_rate=ffn_dropout_rate,
        num_blocks=num_blocks,
        num_heads=num_heads,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=activation_checkpoint,
        verbose=verbose,
    )
    
@gin.configurable
def tisasrec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    num_blocks: int = 2,
    num_heads: int = 1,
    time_span: int = 256,
    dropout_rate: float = 0.2,
    ffn_hidden_dim: int = 64,
) -> GeneralizedInteractionModule:
    return TiSASRec(
        embedding_module=embedding_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        num_blocks=num_blocks,
        time_span=time_span,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=activation_checkpoint,
        verbose=verbose,
    )    

@gin.configurable
def lrurec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    ffn_multiply: int = 1,
    ffn_dropout_rate: float = 0.2,
    num_blocks: int = 2,
) -> GeneralizedInteractionModule:
    return LRURec(
        embedding_module=embedding_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        ffn_multiply=ffn_multiply,
        dropout_rate=ffn_dropout_rate,
        ffn_dropout_rate=ffn_dropout_rate,
        num_blocks=num_blocks,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=activation_checkpoint,
        verbose=verbose,
    )

@gin.configurable
def mamba4rec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    d_state: int,
    d_conv: int,
    expand: int,
    expand_ffn: int,
    num_blocks: int,
    dropout_rate: float,
    use_time_interval_embedding: bool = False,
) -> GeneralizedInteractionModule:
    return Mamba4Rec(
        embedding_module=embedding_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        dropout_rate=dropout_rate,
        num_blocks=num_blocks,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        expand_ffn=expand_ffn,
        use_time_interval_embedding=use_time_interval_embedding,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=activation_checkpoint,
        verbose=verbose,
    )

@gin.configurable
def ttt4rec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    num_blocks: int = 2,
    num_heads: int = 4,
    intermediate_size=256,
    rope_theta=10000.0,
    ttt_layer_type="mlp",
    ttt_base_lr=1.0,
    mini_batch_size=8,
    use_gate=False,
    pre_conv=False,
    share_qk=False,
) -> GeneralizedInteractionModule:
    return TTT4Rec(
        embedding_module=embedding_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        intermediate_size=intermediate_size,
        rope_theta=rope_theta,
        ttt_layer_type=ttt_layer_type,
        ttt_base_lr=ttt_base_lr,
        mini_batch_size=mini_batch_size,
        use_gate=use_gate,
        pre_conv=pre_conv,
        share_qk=share_qk,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=activation_checkpoint,
        verbose=verbose,
    )

@gin.configurable
def hstu_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    num_blocks: int = 2,
    num_heads: int = 1,
    dqk: int = 64,
    dv: int = 64,
    linear_dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    normalization: str = "rel_bias",
    linear_config: str = "uvqk",
    linear_activation: str = "silu",
    concat_ua: bool = False,
    enable_relative_attention_bias: bool = True,
) -> GeneralizedInteractionModule:
    return HSTU(
        embedding_module=embedding_module,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        attention_dim=dqk,
        linear_dim=dv,
        linear_dropout_rate=linear_dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        linear_config=linear_config,
        linear_activation=linear_activation,
        normalization=normalization,
        concat_ua=concat_ua,
        enable_relative_attention_bias=enable_relative_attention_bias,
        verbose=verbose,
    )

@gin.configurable
def tim4rec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    
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
    is_ffn = True,
    is_time = True, 
    p2p_residual = False,
    norm_eps: float = 1e-10,
    is_kai_ming_init: bool = False,
    verbose: bool = False,
) :
    return TiM4Rec(
        embedding_module=embedding_module,
        embedding_dim=embedding_module.item_embedding_dim,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=activation_checkpoint,
        verbose=verbose,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        num_blocks=num_blocks,
        dropout_prob=dropout_prob,
        time_drop_out=time_drop_out,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        head_dim=head_dim,
        chunk_size=chunk_size,
        is_ffn=is_ffn,
        is_time=is_time,
        p2p_residual=p2p_residual,
        norm_eps=norm_eps,
        is_kai_ming_init=is_kai_ming_init,
    )

@gin.configurable
def fuxi_linear_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    channel_t_config = {
        'type': 'linear',
        'num_heads': 8,
        'base': 2,
        'start_index': 0,
    },
    channel_p_config = {
        'type': 'linear',
        'dim': 32,
        'aug_current': True,
    },
    use_rope: bool = False, 
    num_blocks: int = 2,
    num_heads: int = 1,
    dqk: int = 64,
    dv: int = 64,
    chunk_size: Optional[int] = None,
    linear_dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    ffn_multiply: int = 1,
    normalization: str = "rel_bias",
    linear_activation: str = "silu",
    enable_relative_attention_bias: bool = True,
) -> GeneralizedInteractionModule:
    return FuXiLinear(
        embedding_module=embedding_module,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        channel_t_config=channel_t_config,
        channel_p_config=channel_p_config,
        use_rope=use_rope,
        num_blocks=num_blocks,
        num_heads=num_heads,
        chunk_size=chunk_size,
        attention_dim=dqk,
        linear_dim=dv,
        linear_dropout_rate=linear_dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        ffn_multiply=ffn_multiply,
        linear_activation=linear_activation,
        normalization=normalization,
        enable_relative_attention_bias=enable_relative_attention_bias,
        verbose=verbose,
    )

@gin.configurable
def fuxi_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    num_blocks: int = 2,
    num_heads: int = 1,
    dqk: int = 64,
    dv: int = 64,
    linear_dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    ffn_multiply: int = 1,
    ffn_single_stage: bool = False,
    normalization: str = "rel_bias",
    linear_config: str = "uvqk",
    linear_activation: str = "silu",
    enable_relative_attention_bias: bool = True,
) -> GeneralizedInteractionModule:
    return FuXi(
        embedding_module=embedding_module,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        attention_dim=dqk,
        linear_dim=dv,
        linear_dropout_rate=linear_dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        ffn_multiply=ffn_multiply,
        ffn_single_stage=ffn_single_stage,
        linear_config=linear_config,
        linear_activation=linear_activation,
        normalization=normalization,
        enable_relative_attention_bias=enable_relative_attention_bias,
        verbose=verbose,
    )

@gin.configurable
def fuxi_beta_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    num_blocks: int = 2,
    num_heads: int = 1,
    dqk: int = 64,
    dv: int = 64,
    linear_dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    ffn_multiply: int = 1,
    func_type: str = 'pow',
    normalization: str = "rel_bias",
    linear_activation: str = "silu",
    enable_relative_attention_bias: bool = True,
) -> GeneralizedInteractionModule:
    return FuXiBeta(
        embedding_module=embedding_module,
        similarity_module=interaction_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        attention_dim=dqk,
        linear_dim=dv,
        linear_dropout_rate=linear_dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        ffn_multiply=ffn_multiply,
        func_type=func_type,
        linear_activation=linear_activation,
        normalization=normalization,
        enable_relative_attention_bias=enable_relative_attention_bias,
        verbose=verbose,
    )

@gin.configurable
def get_sequential_encoder(
    module_type: str,
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    verbose: bool,
    activation_checkpoint: bool = False,
) -> GeneralizedInteractionModule:
    module_dict = {
        'Fuxi-Linear': fuxi_linear_encoder,  
        'lrurec': lrurec_encoder,
        'mamba4rec': mamba4rec_encoder,
        'tim4rec': tim4rec_encoder,
        'TiSASRec': tisasrec_encoder,
        'ttt4rec': ttt4rec_encoder,
    }
    
    if module_type == "SASRec":
        model = sasrec_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    elif module_type == "HSTU":
        model = hstu_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    elif module_type == 'FuXi' :
        model = fuxi_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    elif module_type == 'FuXi-Beta' :
        model = fuxi_beta_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    elif module_type in module_dict :
        module_func = module_dict[module_type]
        model = module_func(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported module_type {module_type}")
    return model
