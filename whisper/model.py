import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import tensorflow as tf
from whisper.decoder import decode as decode_function
from whisper.decoder import beam_search_decode as decode_function
from .transcribe import transcribe as transcribe_function

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


class LayerNorm(tf.keras.layers.LayerNormalization):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_float = tf.cast(x, tf.float32)
        normed = super().call(x_float)
        x_original = tf.cast(normed, x.dtype)
        return x_original

    
class Linear(tf.keras.layers.Dense):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        weights = tf.cast(self.kernel, x.dtype)
        biases = tf.cast(self.bias, x.dtype) if self.use_bias else None

        output = tf.matmul(x, weights)
        if biases is not None:
            output = tf.nn.bias_add(output, biases)

        return output


class Conv1d(tf.keras.layers.Conv1D):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        weights = tf.cast(self.kernel, x.dtype)
        biases = tf.cast(self.bias, x.dtype) if self.use_bias else None

        output = tf.nn.conv1d(x, weights, self.strides[0], self.padding.upper())
        if biases is not None:
            output = tf.nn.bias_add(output, biases)

        return output

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


class MultiHeadAttention(tf.keras.layers.Layer):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state)
        self.key = Linear(n_state, use_bias=False)
        self.value = Linear(n_state)
        self.out = Linear(n_state)


    def call(
        self,
        x: tf.Tensor,
        xa: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:

            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

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

        q = q * scale
        k = k * scale
        qk = tf.matmul(q, k, transpose_b=True)

        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = tf.cast(qk, tf.float32)

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


class ResidualAttentionBlock(tf.keras.layers.Layer):

    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(axis=-1)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(axis=-1) if cross_attention else None

        self.mlp = self.mlp = MLP(n_state)
        self.mlp_ln = LayerNorm(axis=-1)

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
            cross_attn_out, _ = self.cross_attn(x=self.cross_attn_ln(x),xa=xa,mask=None,kv_cache=kv_cache)

            x = x + cross_attn_out

        x = x + self.mlp(self.mlp_ln(x))
        return x
    
    

class AudioEncoder(tf.keras.layers.Layer):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(filters=n_state, kernel_size=3, strides=1, padding='same')
        self.conv2 = Conv1d(filters=n_state, kernel_size=3, strides=2, padding='same')

        self.positional_embedding = sinusoids(n_ctx, n_state)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = LayerNorm(axis=-1)

    def call(self, x: tf.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = tf.nn.gelu(self.conv1(x))
        x = tf.nn.gelu(self.conv2(x))
        seq_len = tf.shape(x)[1]
        pos_emb = self.positional_embedding[:seq_len]
        x = x + tf.cast(pos_emb, x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
class TextDecoder(tf.keras.layers.Layer):
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

        # Create causal mask: prevent attention to future tokens
        mask = 1.0 - tf.linalg.band_part(tf.ones((n_ctx, n_ctx)), -1, 0)
        mask = tf.where(mask == 1.0, tf.constant(-1e9, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        self.causal_mask = tf.constant(mask, dtype=tf.float32)

    def call(self, x: tf.Tensor, xa: tf.Tensor, kv_cache: Optional[dict] = None):
        """
        x : tf.Tensor, shape = (batch_size, <= n_ctx)
            input text tokens
        xa : tf.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            encoded audio features
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        pos_embed_slice = self.positional_embedding[offset : offset + tf.shape(x)[-1]]
        x = self.token_embedding(x) + pos_embed_slice
        x = tf.cast(x, xa.dtype) 


        for idx, block in enumerate(self.blocks):
            x = block(x, xa, mask=self.causal_mask, kv_cache=kv_cache)

        x = self.ln(x)

        transpose_embedding = tf.transpose(tf.cast(self.token_embedding.weights[0], x.dtype), [1, 0])
        logits = tf.matmul(x, transpose_embedding)
        logits = tf.cast(logits, tf.float32)

        return logits



class Whisper(tf.keras.Model):
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
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = tf.sparse.from_dense(tf.convert_to_tensor(all_heads))

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask =tf.convert_to_tensor(array.reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        ))
        self.alignment_heads = tf.sparse.from_dense(mask)

    def embed_audio(self, mel: tf.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: tf.Tensor, audio_features: tf.Tensor):
        return self.decoder(tokens, audio_features)

    def call(
        self, mel: tf.Tensor, tokens: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    transcribe = transcribe_function
    decode = decode_function
