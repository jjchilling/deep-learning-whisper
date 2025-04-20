import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf
# import torch
# import torch.nn.functional as F
# from torch import Tensor, nn

from whisper.decoder import decode as decode_function
#from .decoding import decode as decode_function
from whisper.decoder import greedy_decode as decode_function

from .transcribe import transcribe as transcribe_function

# try:
#     from torch.nn.functional import scaled_dot_product_attention

#     SDPA_AVAILABLE = True
# except (ImportError, RuntimeError, OSError):
#     scaled_dot_product_attention = None
SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


# class LayerNorm(nn.LayerNorm):
#     def forward(self, x: Tensor) -> Tensor:
#         return super().forward(x.float()).type(x.dtype)

class LayerNorm(tf.keras.layers.LayerNormalization):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_float = tf.cast(x, tf.float32)
        normed = super().call(x_float)
        x_original = tf.cast(normed, x.dtype)
        return x_original

# class Linear(nn.Linear):
#     def forward(self, x: Tensor) -> Tensor:
#         return F.linear(
#             x,
#             self.weight.to(x.dtype),
#             None if self.bias is None else self.bias.to(x.dtype),
#         )
    
class Linear(tf.keras.layers.Dense):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        weights = tf.cast(self.kernel, x.dtype)
        biases = tf.cast(self.bias, x.dtype) if self.use_bias else None

        output = tf.matmul(x, weights)
        if biases is not None:
            output = tf.nn.bias_add(output, biases)

        return output


# class Conv1d(nn.Conv1d):
#     def _conv_forward(
#         self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
#     ) -> Tensor:
#         return super()._conv_forward(
#             x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
#         )

class Conv1d(tf.keras.layers.Conv1D):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        weights = tf.cast(self.kernel, x.dtype)
        biases = tf.cast(self.bias, x.dtype) if self.use_bias else None

        output = tf.nn.conv1d(x, weights, self.strides[0], self.padding.upper())
        if biases is not None:
            output = tf.nn.bias_add(output, biases)

        return output

# def sinusoids(length, channels, max_timescale=10000):
#     """Returns sinusoids for positional embedding"""
#     assert channels % 2 == 0
#     log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
#     inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
#     scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
#     return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    length = tf.cast(length, tf.float32)
    channels = tf.cast(channels, tf.float32)
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = tf.exp(-log_timescale_increment * tf.range(channels // 2))
    scaled_time = tf.range(length)[:, tf.newaxis] * inv_timescales[tf.newaxis, :]
    return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

# no sdpa equivalent in tf (could implement it from scratch??)
# https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/
@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


# class MultiHeadAttention(nn.Module):
#     use_sdpa = True

#     def __init__(self, n_state: int, n_head: int):
#         super().__init__()
#         self.n_head = n_head
#         self.query = Linear(n_state, n_state)
#         self.key = Linear(n_state, n_state, bias=False)
#         self.value = Linear(n_state, n_state)
#         self.out = Linear(n_state, n_state)

class MultiHeadAttention(tf.keras.layers.Layer):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state)
        self.key = Linear(n_state, use_bias=False)
        self.value = Linear(n_state)
        self.out = Linear(n_state)

    # def forward(
    #     self,
    #     x: Tensor,
    #     xa: Optional[Tensor] = None,
    #     mask: Optional[Tensor] = None,
    #     kv_cache: Optional[dict] = None,
    # ):
    #     q = self.query(x)

    #     if kv_cache is None or xa is None or self.key not in kv_cache:
    #         # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
    #         # otherwise, perform key/value projections for self- or cross-attention as usual.
    #         k = self.key(x if xa is None else xa)
    #         v = self.value(x if xa is None else xa)
    #     else:
    #         # for cross-attention, calculate keys and values once and reuse in subsequent calls.
    #         k = kv_cache[self.key]
    #         v = kv_cache[self.value]

    #     wv, qk = self.qkv_attention(q, k, v, mask)
    #     return self.out(wv), qk

    def call(
        self,
        x: tf.Tensor,
        xa: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    # def qkv_attention(
    #     self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     n_batch, n_ctx, n_state = q.shape
    #     scale = (n_state // self.n_head) ** -0.25
    #     q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    #     k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    #     v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    #     if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
    #         a = scaled_dot_product_attention(
    #             q, k, v, is_causal=mask is not None and n_ctx > 1
    #         )
    #         out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
    #         qk = None
    #     else:
    #         qk = (q * scale) @ (k * scale).transpose(-1, -2)
    #         if mask is not None:
    #             qk = qk + mask[:n_ctx, :n_ctx]
    #         qk = qk.float()

    #         w = F.softmax(qk, dim=-1).to(q.dtype)
    #         out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
    #         qk = qk.detach()

    #     return out, qk

    def qkv_attention(
        self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = tf.reshape(q, [n_batch, n_ctx, self.n_head, - 1])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.reshape(k, [n_batch, k.shape[1], self.n_head, -1])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [n_batch, v.shape[1], self.n_head, -1])
        v = tf.transpose(v, [0, 2, 1, 3])

        # if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
        #     a = scaled_dot_product_attention(
        #         q, k, v, is_causal=mask is not None and n_ctx > 1
        #     )
        #     out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        #     qk = None
        # else:
        q = q * scale
        k = k * scale
        qk = tf.matmul(q, k, transpose_b=True)

        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        # qk = qk.float()

        w = tf.cast(tf.nn.softmax(qk, axis=-1), q.dtype)
        wv = tf.matmul(w,v)
        wv = tf.transpose(wv, [0, 2, 1, 3])
        out = tf.reshape(wv, [n_batch, n_ctx, n_state])

        return out, qk

class MLP(tf.keras.layers.Layer):
    def __init__(self, n_state):
        super().__init__()
        self.dense1 = Linear(n_state * 4)
        self.gelu = tf.keras.layers.Activation('gelu')
        self.dense2 = Linear(n_state)

    def call(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x


# class ResidualAttentionBlock(nn.Module):
class ResidualAttentionBlock(tf.keras.layers.Layer):

#     def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
#         super().__init__()

#         self.attn = MultiHeadAttention(n_state, n_head)
#         self.attn_ln = LayerNorm(n_state)

#         self.cross_attn = (
#             MultiHeadAttention(n_state, n_head) if cross_attention else None
#         )
#         self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

#         n_mlp = n_state * 4
#         self.mlp = nn.Sequential(
#             Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
#         )
#         self.mlp_ln = LayerNorm(n_state)
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(axis=-1)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(axis=-1) if cross_attention else None

        self.mlp = self.mlp = MLP(n_state)
        self.mlp_ln = LayerNorm(axis=-1)

#     def forward(
#         self,
#         x: Tensor,
#         xa: Optional[Tensor] = None,
#         mask: Optional[Tensor] = None,
#         kv_cache: Optional[dict] = None,
#     ):
#         x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
#         if self.cross_attn:
#             x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
#         x = x + self.mlp(self.mlp_ln(x))
#         return x
    def call(
        self,
        x: tf.Tensor,
        xa: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        
        attn_out, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        x = x + attn_out
        if self.cross_attn is not None:
            cross_attn_out, _ = self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
            x = x + cross_attn_out

        x = x + self.mlp(self.mlp_ln(x))
        return x
    
    

# class AudioEncoder(nn.Module):
class AudioEncoder(tf.keras.layers.Layer):
#     def __init__(
#         self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
#     ):
#         super().__init__()
#         self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
#         self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
#         self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

#         self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
#             [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
#         )
#         self.ln_post = LayerNorm(n_state)
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_state, kernel_size=3, strides=1, padding='same')
        self.conv2 = Conv1d(n_state, kernel_size=3, strides=2, padding='valid')
        self.positional_embeding = sinusoids(n_ctx, n_state)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = LayerNorm(axis=-1)

#     def forward(self, x: Tensor):
#         """
#         x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
#             the mel spectrogram of the audio
#         """
#         x = F.gelu(self.conv1(x))
#         x = F.gelu(self.conv2(x))
#         x = x.permute(0, 2, 1)

#         assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
#         x = (x + self.positional_embedding).to(x.dtype)

#         for block in self.blocks:
#             x = block(x)

#         x = self.ln_post(x)
#         return x
    def call(self, x: tf.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = tf.nn.gelu(self.conv1(x))
        x = tf.nn.gelu(self.conv2(x))

        seq_len = tf.shape(x)[1]
        pos_emb = self.positional_embeding[:seq_len, :] 
        pos_emb = tf.expand_dims(pos_emb, axis=0)        
        x = tf.cast(x + pos_emb, x.dtype)


        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

# class TextDecoder(nn.Module):
class TextDecoder(tf.keras.layers.Layer):

#     def __init__(
#         self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
#     ):
#         super().__init__()

#         self.token_embedding = nn.Embedding(n_vocab, n_state)
#         self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

#         self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
#             [
#                 ResidualAttentionBlock(n_state, n_head, cross_attention=True)
#                 for _ in range(n_layer)
#             ]
#         )
#         self.ln = LayerNorm(n_state)

#         mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
#         self.register_buffer("mask", mask, persistent=False)

    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = tf.keras.layers.Embedding(n_vocab, n_state)
        self.positional_embedding = self.add_weight(
            shape=(n_ctx, n_state),
            initializer="random_normal",
            name="positional_embedding"
        )

        self.blocks = [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        self.ln = LayerNorm(axis=-1)

        # this is super sus highkey
        inf_fill = tf.fill((n_ctx, n_ctx), -np.inf)
        mask = np.triu(inf_fill, k=1)
        self.mask = self.add_weight(
            shape=(n_ctx, n_ctx),
            initializer=tf.constant_initializer(mask),
            trainable=False,
            name="mask"
        )


#     def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
#         """
#         x : torch.LongTensor, shape = (batch_size, <= n_ctx)
#             the text tokens
#         xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
#             the encoded audio features to be attended on
#         """
#         offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
#         x = (
#             self.token_embedding(x)
#             + self.positional_embedding[offset : offset + x.shape[-1]]
#         )
#         x = x.to(xa.dtype)

#         for block in self.blocks:
#             x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

#         x = self.ln(x)
#         logits = (
#             x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
#         ).float()

#         return logits
    def call(self, x: tf.Tensor, xa: tf.Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        # highkey this is super sus too
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + tf.shape(x)[-1]]
        )
        x = tf.cast(x, xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        transpose_embedding = tf.transpose(tf.cast(self.token_embedding.weights[0], x.dtype), [1,0])
        logits = tf.matmul(x, transpose_embedding)
        logits = tf.cast(logits, tf.float32)

        return logits


# class Whisper(nn.Module):
class Whisper(tf.keras.Model):
    # def __init__(self, dims: ModelDimensions):
    #     super().__init__()
    #     self.dims = dims
    #     self.encoder = AudioEncoder(
    #         self.dims.n_mels,
    #         self.dims.n_audio_ctx,
    #         self.dims.n_audio_state,
    #         self.dims.n_audio_head,
    #         self.dims.n_audio_layer,
    #     )
    #     self.decoder = TextDecoder(
    #         self.dims.n_vocab,
    #         self.dims.n_text_ctx,
    #         self.dims.n_text_state,
    #         self.dims.n_text_head,
    #         self.dims.n_text_layer,
    #     )
    #     # use the last half among the decoder layers for time alignment by default;
    #     # to use a specific set of heads, see `set_alignment_heads()` below.
    #     all_heads = torch.zeros(
    #         self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
    #     )
    #     all_heads[self.dims.n_text_layer // 2 :] = True
    #     self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = tf.sparse.from_dense(tf.convert_to_tensor(all_heads))

    # def set_alignment_heads(self, dump: bytes):
    #     array = np.frombuffer(
    #         gzip.decompress(base64.b85decode(dump)), dtype=bool
    #     ).copy()
    #     mask = torch.from_numpy(array).reshape(
    #         self.dims.n_text_layer, self.dims.n_text_head
    #     )
    #     self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)
    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask =tf.convert_to_tensor(array.reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        ))
        self.alignment_heads = tf.sparse.from_dense(mask)
        # self.add_weight(name='alignment_heads',
        #                 shape=
        #                 initializer=tf.initself.alignment_heads, trainable=False)

    # def embed_audio(self, mel: torch.Tensor):
    #     return self.encoder(mel)
    def embed_audio(self, mel: tf.Tensor):
        return self.encoder(mel)

    # def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
    #     return self.decoder(tokens, audio_features)
    def logits(self, tokens: tf.Tensor, audio_features: tf.Tensor):
        return self.decoder(tokens, audio_features)

    # def forward(
    #     self, mel: torch.Tensor, tokens: torch.Tensor
    # ) -> Dict[str, torch.Tensor]:
    #     return self.decoder(tokens, self.encoder(mel))
    def call(
        self, mel: tf.Tensor, tokens: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    # highkey what is going on here?????
    # @property
    # def device(self):
    #     return next(self.parameters()).device

    # @property
    # def is_multilingual(self):
    #     return self.dims.n_vocab >= 51865
    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    # @property
    # def num_languages(self):
    #     return self.dims.n_vocab - 51765 - int(self.is_multilingual)
    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    # def install_kv_cache_hooks(self, cache: Optional[dict] = None):
    #     """
    #     The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
    #     tensors calculated for the previous positions. This method returns a dictionary that stores
    #     all caches, and the necessary hooks for the key and value projection modules that save the
    #     intermediate tensors to be reused during later calculations.

    #     Returns
    #     -------
    #     cache : Dict[nn.Module, torch.Tensor]
    #         A dictionary object mapping the key/value projection modules to its cache
    #     hooks : List[RemovableHandle]
    #         List of PyTorch RemovableHandle objects to stop the hooks to be called
    #     """
    #     cache = {**cache} if cache is not None else {}
    #     hooks = []
    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        # def save_to_cache(module, _, output):
        #     if module not in cache or output.shape[1] > self.dims.n_text_ctx:
        #         # save as-is, for the first token or cross attention
        #         cache[module] = output
        #     else:
        #         cache[module] = torch.cat([cache[module], output], dim=1).detach()
        #     return cache[module]
        
        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        # def install_hooks(layer: nn.Module):
        #     if isinstance(layer, MultiHeadAttention):
        #         hooks.append(layer.key.register_forward_hook(save_to_cache))
        #         hooks.append(layer.value.register_forward_hook(save_to_cache))
        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        # self.decoder.apply(install_hooks)
        # return cache, hooks
        self.decoder.apply(install_hooks)
        return cache, hooks

    transcribe = transcribe_function
    decode = decode_function
