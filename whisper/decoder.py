import tensorflow as tf
import numpy as np

class DecodingOptions:
    def __init__(self, beam_size=None, temperature=0.0, max_len=128, suppress_blank=True, suppress_tokens="-1", use_timestamps=False, max_initial_timestamp=None):
        self.beam_size = beam_size
        self.temperature = temperature
        self.max_len = max_len
        self.suppress_blank = suppress_blank
        self.suppress_tokens = suppress_tokens
        self.use_timestamps = use_timestamps
        self.max_initial_timestamp = max_initial_timestamp

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_len, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = self.add_weight(
            shape=(max_len, embed_dim), initializer="random_normal", trainable=True, name="pos_embedding"
        )

    def call(self, x):
        max_len = tf.shape(x)[-1]
        return self.token_emb(x) + self.pos_emb[:max_len]

def mlp_block(embed_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(4 * embed_dim, activation=tf.keras.activations.gelu),
        tf.keras.layers.Dense(embed_dim)
    ])

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mlp = mlp_block(embed_dim)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.ln3 = tf.keras.layers.LayerNormalization()

    def call(self, x, audio_features, causal_mask=None):
        x = x + self.self_attn(query=self.ln1(x), value=x, key=x, attention_mask=causal_mask)
        x = x + self.cross_attn(query=self.ln2(x), value=audio_features, key=audio_features)
        x = x + self.mlp(self.ln3(x))
        return x

class WhisperDecoder(tf.keras.Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(vocab_size, max_len, embed_dim)
        self.blocks = [DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)]
        self.ln = tf.keras.layers.LayerNormalization()
        self.output_proj = tf.keras.layers.Dense(vocab_size)

    def call(self, tokens, audio_features, kv_cache=None):
        x = self.embedding(tokens)
        seq_len = tf.shape(tokens)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.reshape(causal_mask, [1, 1, seq_len, seq_len])

        for i, block in enumerate(self.blocks):
            x = block(x, audio_features, causal_mask=causal_mask)

        x = self.ln(x)
        return self.output_proj(x)

class LogitFilter:
    def apply(self, logits, tokens):
        pass

class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer, sample_begin):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits, tokens):
        if tf.shape(tokens)[1] == self.sample_begin:
            for s in self.tokenizer.encode(" ") + [self.tokenizer.eot]:
                logits = tf.tensor_scatter_nd_update(
                    logits, indices=[[i, s] for i in range(tf.shape(logits)[0])], updates=[-1e9] * tf.shape(logits)[0]
                )
        return logits

class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits, tokens):
        for s in self.suppress_tokens:
            logits = tf.tensor_scatter_nd_update(
                logits, indices=[[i, s] for i in range(tf.shape(logits)[0])], updates=[-1e9] * tf.shape(logits)[0]
            )
        return logits

class ApplyTimestampRules(LogitFilter):
    def __init__(self, tokenizer, timestamp_begin=50364):
        self.tokenizer = tokenizer
        self.timestamp_begin = timestamp_begin

    def apply(self, logits, tokens):
        seq = tokens.numpy().tolist()
        last_token = seq[0][-1] if seq[0] else None
        penultimate_token = seq[0][-2] if len(seq[0]) >= 2 else None

        if last_token is not None and last_token >= self.timestamp_begin:
            if penultimate_token is not None and penultimate_token >= self.timestamp_begin:
                logits = tf.tensor_scatter_nd_update(
                    logits, indices=[[i, j] for i in range(logits.shape[0]) for j in range(self.timestamp_begin, logits.shape[1])],
                    updates=[-1e9] * logits.shape[0] * (logits.shape[1] - self.timestamp_begin)
                )
            else:
                logits = tf.tensor_scatter_nd_update(
                    logits, indices=[[i, j] for i in range(logits.shape[0]) for j in range(0, self.timestamp_begin)],
                    updates=[-1e9] * logits.shape[0] * self.timestamp_begin
                )
        return logits

class DecodingResult:
    def __init__(self, tokens, text):
        self.tokens = tokens
        self.text = text

class DecodingTask:
    def __init__(self, encoder, decoder, tokenizer, options: DecodingOptions):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.options = options
        self.logit_filters = []
        self.sample_begin = 1

        print("time to decode")

        if options.use_timestamps:
            self.logit_filters.append(ApplyTimestampRules(tokenizer))
        if options.suppress_blank:
            self.logit_filters.append(SuppressBlank(tokenizer, self.sample_begin))
        if options.suppress_tokens:
            suppress = tokenizer.non_speech_tokens if options.suppress_tokens == "-1" else list(map(int, options.suppress_tokens.split(",")))
            self.logit_filters.append(SuppressTokens(suppress))

    def apply_filters(self, logits, tokens):
        for filt in self.logit_filters:
            logits = filt.apply(logits, tokens)
        return logits

    def _main_loop(self, audio_features, tokens):
        for _ in range(self.options.max_len):
            logits = self.decoder(tokens, audio_features)
            logits = logits[:, -1, :]
            logits = self.apply_filters(logits, tokens)
            next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
            tokens = tf.concat([tokens, tf.expand_dims(next_token, axis=1)], axis=-1)
            if tf.reduce_all(tf.equal(next_token, self.tokenizer.eot)):
                break
        return tokens

    def run(self, mel):
        mel = tf.expand_dims(mel, axis=0)
        mel = tf.transpose(mel, [0, 2, 1])
        audio_features = self.encoder(mel)
        tokens = tf.constant([[self.tokenizer.sot]], dtype=tf.int32)

        if self.options.beam_size:
            token_ids = beam_search_decode(self.decoder, audio_features, self.tokenizer, self.apply_filters, beam_size=self.options.beam_size, max_len=self.options.max_len)
        else:
            token_ids = self._main_loop(audio_features, tokens)

        text = self.tokenizer.decode(token_ids[0].numpy().tolist())
        return DecodingResult(token_ids[0], text)

def transcribe_from_mel(mel, encoder, decoder, tokenizer, beam_size=None):
    options = DecodingOptions(beam_size=beam_size)
    task = DecodingTask(encoder, decoder, tokenizer, options)
    return task.run(mel)

def decode(encoder, decoder, tokenizer, mel, options=DecodingOptions()):
    if len(mel.shape) == 2:
        mel = tf.expand_dims(mel, axis=0)
        print("expanded dimensions")
    task = DecodingTask(encoder, decoder, tokenizer, options)
    return task.run(mel)

def greedy_decode(decoder, audio_features, tokenizer, apply_filters, max_len=128):
    tokens = tf.constant([[tokenizer.sot]], dtype=tf.int32)
    for _ in range(max_len):
        logits = decoder(tokens, audio_features)
        logits = logits[:, -1, :]
        logits = apply_filters(logits, tokens)
        next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
        tokens = tf.concat([tokens, tf.expand_dims(next_token, axis=1)], axis=-1)
        if tf.reduce_all(tf.equal(next_token, tokenizer.eot)):
            break
    return tokens

def beam_search_decode(decoder, audio_features, tokenizer, apply_filters, beam_size=5, max_len=128):
    sequences = [(tf.constant([[tokenizer.sot]], dtype=tf.int32), 0.0)]

    for _ in range(max_len):
        all_candidates = []
        for tokens, score in sequences:
            logits = decoder(tokens, audio_features)
            logits = logits[:, -1, :]
            logits = apply_filters(logits, tokens)
            log_probs = tf.nn.log_softmax(logits)
            topk_log_probs, topk_tokens = tf.math.top_k(log_probs, k=beam_size)

            for i in range(beam_size):
                new_token = topk_tokens[0, i]
                new_score = score + float(topk_log_probs[0, i])
                new_seq = tf.concat([tokens, tf.constant([[new_token]])], axis=-1)
                all_candidates.append((new_seq, new_score))

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_size]

        if all(tf.reduce_all(tf.equal(seq[0][0, -1], tokenizer.eot)) for seq in sequences):
            break

    return sequences[:1][0][0]
