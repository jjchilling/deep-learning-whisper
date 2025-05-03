import tensorflow as tf

class DecodingOptions:
    def __init__(self, beam_size=None, temperature=0.0, max_len=128, suppress_blank=True, suppress_tokens="-1", use_timestamps=False, max_initial_timestamp=None):
        self.beam_size = beam_size
        self.temperature = temperature
        self.max_len = max_len
        self.suppress_blank = suppress_blank
        self.suppress_tokens = suppress_tokens
        self.use_timestamps = use_timestamps
        self.max_initial_timestamp = max_initial_timestamp

class LogitFilter:
    def apply(self, logits, tokens):
        pass

class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer, sample_begin):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits, tokens):
        def suppress_logits(logits, suppressed_ids):
            batch_size = tf.shape(logits)[0]
            indices = tf.stack([
                tf.repeat(tf.range(batch_size), len(suppressed_ids)),
                tf.tile(tf.constant(suppressed_ids, dtype=tf.int32), [batch_size])
            ], axis=1)
            updates = tf.fill([batch_size * len(suppressed_ids)], -1e9)
            return tf.tensor_scatter_nd_update(logits, indices, updates)

        if tf.shape(tokens)[1] == self.sample_begin:
            logits = suppress_logits(logits, [self.tokenizer.eot])
        return logits


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits, tokens):
        def suppress_logits(logits, suppressed_ids):
            batch_size = tf.shape(logits)[0]
            indices = tf.stack([
                tf.repeat(tf.range(batch_size), len(suppressed_ids)),
                tf.tile(tf.constant(suppressed_ids, dtype=tf.int32), [batch_size])
            ], axis=1)
            updates = tf.fill([batch_size * len(suppressed_ids)], -1e9)
            return tf.tensor_scatter_nd_update(logits, indices, updates)

        logits = suppress_logits(logits, self.suppress_tokens)
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
        
# class RepetitionPenalty(LogitFilter):
#     def __init__(self, penalty: float = 1.2):
#         self.penalty = penalty

#     def apply(self, logits: tf.Tensor, tokens: tf.Tensor) -> tf.Tensor:
#         for b in range(logits.shape[0]):
#             for t in tokens[b]:
#                 logits = tf.tensor_scatter_nd_update(logits,[[b, t]],[logits[b, t] / self.penalty])
#         return logits
    
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
        # self.logit_filters.append(RepetitionPenalty(penalty=1.2))

    def apply_filters(self, logits, tokens):
        for filt in self.logit_filters:
            logits = filt.apply(logits, tokens)
        return logits

    def greedy_decode(self, audio_features, tokens,temperature=1.0):
        for _ in range(self.options.max_len):
            logits = self.decoder(tokens, audio_features)
            logits = logits[:, -1, :]
            logits = self.apply_filters(logits, tokens)
            if temperature != 1.0:
                logits = logits / temperature
            next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
            tokens = tf.concat([tokens, tf.expand_dims(next_token, axis=1)], axis=-1)
            if tf.reduce_all(tf.equal(next_token, self.tokenizer.eot)):
                break
        return tokens

    def run(self, mel, model):
        if len(mel.shape) == 2:
            mel = tf.expand_dims(mel, axis=0)

        decoded = [self.tokenizer.sot]
        for _ in range(100):
            decoder_input = tf.expand_dims(tf.constant(decoded, dtype=tf.int32), axis=0)
            logits = model(mel, decoder_input)
            next_token = tf.argmax(logits[:, -1, :], axis=-1).numpy()[0]
            decoded.append(next_token)
            if next_token == self.tokenizer.eot:
                break

        text_pred = self.tokenizer.decode(decoded)
        # audio_features = self.encoder(mel)
        # tokens = tf.constant([[self.tokenizer.sot]], dtype=tf.int32)

        # if self.options.beam_size:
        #     token_ids = beam_search_decode(self.decoder, audio_features, self.tokenizer, self.apply_filters, beam_size=self.options.beam_size, max_len=self.options.max_len)
        # else:
        #     token_ids = self.greedy_decode(audio_features, tokens,self.options.temperature)

        # text = self.tokenizer.decode(token_ids[0].numpy().tolist())
        return DecodingResult(decoded, text_pred)

def transcribe_from_mel(mel, encoder, decoder, tokenizer, beam_size=None):
    options = DecodingOptions(beam_size=beam_size)
    task = DecodingTask(encoder, decoder, tokenizer, options)
    return task.run(mel, model)

def decode(encoder, decoder, tokenizer, mel, mode, options=DecodingOptions()):
    if len(mel.shape) == 2:
        mel = tf.expand_dims(mel, axis=0)
    task = DecodingTask(encoder, decoder, tokenizer, options)
    return task.run(mel,mode)


def beam_search_decode(decoder, audio_features, tokenizer, apply_filters, beam_size=5, max_len=128):
    sequences = [(tf.constant([tokenizer.sot_sequence], dtype=tf.int32), 0.0)]

    for step in range(max_len):
        all_candidates = []

        for tokens, score in sequences:
            logits = decoder(tokens, audio_features)
            logits = logits[:, -1, :]
            logits = apply_filters(logits, tokens)
            log_probs = tf.nn.log_softmax(logits)

            topk_log_probs, topk_tokens = tf.math.top_k(log_probs, k=beam_size)

            for i in range(beam_size):
                next_token = topk_tokens[0, i]
                next_score = score + float(topk_log_probs[0, i].numpy())
                new_seq = tf.concat([tokens, tf.expand_dims(next_token, axis=0)], axis=0) 
                all_candidates.append((new_seq, next_score))

        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

        if all(seq[-1].numpy() == tokenizer.eot for seq, _ in sequences):
            break


    return tf.expand_dims(sequences[0][0], axis=0)

