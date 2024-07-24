""" PyTorch ChatGLM model. """

import math

import torch.utils.checkpoint
from typing import Optional, Tuple, Union, List, Iterable

from transformers.modeling_outputs import BaseModelOutputWithPast

import torch
from torch import nn
from argparse import Namespace
import torch.nn.functional as F
from transformers.activations import ACT2FN
from torch.nn import LayerNorm
from torchvision import transforms
from PIL import Image

from vllm.transformers_utils.configs.chatglm import ChatGLMConfig
from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.chatglm import GLMTransformer
from vllm.multimodal.image import (
    cached_get_tokenizer,
    repeat_and_pad_image_tokens,
)
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceData


from .clip import dummy_seq_data_for_clip
from .interfaces import SupportsVision

IMAGE_FEATURE_SIZE = 1602
IMAGE_TOKEN_ID = 151339


def standard_attention(
    query_layer, key_layer, value_layer, scaling_attention_score=True
):
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_probs = F.softmax(attention_scores, dim=-1)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer


def attention_fn_default(
    query_layer, key_layer, value_layer, scaling_attention_score=True
):
    if int(torch.__version__.split(".")[0]) >= 2 and scaling_attention_score:
        # Pytorch 2.0 attention uses very much memory if attention_mask is float, and has NaN bug if attention_mask is None.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return attn_output
    else:
        return standard_attention(
            query_layer,
            key_layer,
            value_layer,
            scaling_attention_score=scaling_attention_score,
        )


class VisualPatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(
            config.num_positions, config.hidden_size
        )

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class VisualAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim**-0.5
        self.query_key_value = nn.Linear(
            config.hidden_size, config.hidden_size * 3
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )  # 3, B, H, L, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = attention_fn_default(q, k, v)
        output = self.dense(out.transpose(1, 2).view(B, L, -1))
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn_weights = attn_weights.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class VisualMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class VisualTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = VisualAttention(config)
        self.mlp = VisualMLP(config)
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class VisualTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                VisualTransformerLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(
            in_features, config.hidden_size, bias=False
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.ffn_hidden_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.ffn_hidden_size, bias=False
        )
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = VisualPatchEmbedding(vision_config)
        self.transformer = VisualTransformer(vision_config)
        self.linear_proj = GLU(config, in_features=config.hidden_size)
        self.conv = nn.Conv2d(
            in_channels=vision_config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=2,
            stride=2,
        )
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        x = x / self.scaling_factor
        return x


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            kv_size = (
                config.num_layers
                * config.kv_channels
                * config.multi_query_group_num
                * 2
            )
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size),
            )
        else:
            self.embedding = torch.nn.Embedding(
                config.pre_seq_len,
                config.num_layers
                * config.kv_channels
                * config.multi_query_group_num
                * 2,
            )

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


def dummy_data_for_glm(ctx: InputContext, seq_len: int):
    hf_config = ctx.get_hf_config(ChatGLMConfig)
    if getattr(hf_config, "vision_config", None):
        vision_config = hf_config.vision_config
    else:
        return SequenceData([0] * seq_len), {}
    image_size = vision_config.get("image_size")
    seq_data = dummy_seq_data_for_clip(
        vision_config,
        seq_len,
        image_token_id=IMAGE_TOKEN_ID,
        image_feature_size_override=IMAGE_FEATURE_SIZE,
    )
    mm_data = {"image": Image.new("RGB", (image_size, image_size), color=0)}
    return seq_data, mm_data


def input_processor_for_glm(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer, trust_remote_code=True
    )
    new_prompt, new_token_ids = repeat_and_pad_image_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        image_token_id=IMAGE_TOKEN_ID,
        repeat_count=IMAGE_FEATURE_SIZE,
    )
    return LLMInputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data=multi_modal_data,
    )


class GlmImageProcessor:
    def __init__(self) -> None:
        self.processor = transforms.Compose(
            [
                transforms.Resize(
                    (1120, 1120),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(self, input_content, data):
        return {"pixel_values": self.processor(data)}


@MULTIMODAL_REGISTRY.register_image_input_mapper(GlmImageProcessor())
@MULTIMODAL_REGISTRY.register_max_image_tokens(IMAGE_FEATURE_SIZE)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_glm)
@INPUT_REGISTRY.register_input_processor(input_processor_for_glm)
class ChatGLMModel(nn.Module, SupportsVision):
    def __init__(
        self,
        config: ChatGLMConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config

        init_method = default_init
        init_kwargs = {}

        self.embedding = init_method(
            VocabParallelEmbedding, config.padded_vocab_size, config.hidden_size
        )
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        self.output_layer = ParallelLMHead(
            config.padded_vocab_size, config.hidden_size
        )

        if getattr(config, "vision_config", None):
            self.vision = EVA2CLIPModel(config)
        self.sampler = Sampler()
        self.logits_processor = LogitsProcessor(config.padded_vocab_size)

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(
            self.output_layer, hidden_states, sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def get_input_embeddings(self):
        return self.embedding

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "output_layer" in name:
                self.output_layer.weight_loader(
                    self.output_layer.weight, loaded_weight
                )
                continue
            if name.startswith("transformer.encoder") or name.startswith(
                "transformer.vision"
            ):
                name = name.replace("transformer.", "", 1)
            elif name.startswith("vision."):
                name = name.replace("vision.", "vision.transformer.", 1)
            if "word_embeddings" in name:
                name = "embedding.weight"
            if "rotary_pos_emb.inv_freq" in name:
                continue
            if "output" in name:
                continue
            param = params_dict[name]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            weight_loader(param, loaded_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, position_ids and (attention_mask = None is fine)"""

        pixel_values = kwargs.pop("pixel_values", None)
        image_features = kwargs.pop("image_features", None)
        if image_features is not None and pixel_values is None:
            pixel_values = image_features
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.output_layer.weight.dtype)
            # this model has two image placeholder 151339 <|begin_of_image|> 151340 <|end_of_image|>
            # we use the 151339 as the image placeholder
            image_token_mask = input_ids == IMAGE_TOKEN_ID
            image_embeds = self.vision(pixel_values.reshape(-1, 3, 1120, 1120))
            inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[image_token_mask] = image_embeds.reshape(
                -1, self.config.hidden_size
            )
        else:
            if input_ids is not None:
                # When there is only text input and no image input
                inputs_embeds = self.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = None

        hidden_states = self.encoder(
            inputs_embeds,
            positions,
            kv_caches,
            attn_metadata,
        )

        return hidden_states
