import os
import random
import numpy as np
import tensorflow as tf
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import get_tokenizer
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from tqdm import tqdm

# ---- CONFIGURATION ----
DATA_DIR = "/Users/robertogonzales/Desktop/DL/WhisperData/LibriSpeech/dev-clean"
EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 1

# ---- FUNCTIONS ----
def split_dataset(dev_dir, split_ratio=0.8):
    all_samples = []
    for root, _, files in os.walk(dev_dir):
        trans_files = [f for f in files if f.endswith('.trans.txt')]
        for trans_file in trans_files:
            with open(os.path.join(root, trans_file), 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utt_id, transcription = parts
                        audio_path = os.path.join(root, utt_id + ".flac")
                        if os.path.exists(audio_path):
                            all_samples.append((audio_path, transcription))
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * split_ratio)
    return all_samples[:split_idx], all_samples[split_idx:]

def wer(ref, hyp):
    ref = ref.split()
    hyp = hyp.split()
    mat = np.zeros((len(ref)+1, len(hyp)+1))
    for i in range(len(ref)+1): mat[i][0] = i
    for j in range(len(hyp)+1): mat[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                mat[i][j] = mat[i-1][j-1]
            else:
                mat[i][j] = min(mat[i-1][j-1]+1, mat[i][j-1]+1, mat[i-1][j]+1)
    return mat[len(ref)][len(hyp)] / len(ref)

def test_model(model, tokenizer, train_samples):
    print("\\nEvaluating on training data...")
    for audio_path, transcription in train_samples:
        # Load and process audio
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(pad_or_trim(audio, length=480000))
        mel = pad_or_trim(mel, length=3000, axis=-1)
        mel = tf.expand_dims(tf.transpose(mel), axis=0) 
        
        encoded_audio = model.encoder(mel)
        
        decoded_tokens = [tokenizer.sot]
        for _ in range(148):
            decoder_input = tf.expand_dims(tf.constant(decoded_tokens, dtype=tf.int32), axis=0)
            logits = model.decoder(decoder_input, encoded_audio)
            next_token = tf.argmax(logits[:, -1, :], axis=-1).numpy()[0]
            decoded_tokens.append(next_token)
            if next_token == tokenizer.eot:
                break
        
        predicted_text = tokenizer.decode(decoded_tokens)
        ground_truth = transcription


        print("nGT:", ground_truth)
        print("PR: ", predicted_text)
        print("WER: ", wer(ground_truth,predicted_text))

    print("n Done evaluating!")


# ---- MAIN ----
def main():
    dims = ModelDimensions(
        n_mels=80, n_audio_ctx=3000, n_audio_state=384, n_audio_head=6, n_audio_layer=4,
        n_vocab=50268, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4
    )
    model = Whisper(dims)
    tokenizer = get_tokenizer()

    print("Splitting dataset...")
    train_samples, testing = split_dataset(DATA_DIR)

    print("Preparing data...")
    data = []
    for audio_path, transcription in train_samples:
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(pad_or_trim(audio, length=480000))
        mel = pad_or_trim(mel, length=3000, axis=-1)
        tokens = tokenizer.encode(transcription)
        tokens.append(tokenizer.eot)
        data.append((mel, tokens))

    def gen():
        for mel, tokens in data:
            yield mel, tokens

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(80, 3000), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).padded_batch(BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        progbar = tqdm(dataset, desc="Training", unit="batch")
        for step, (mel, target_tokens) in enumerate(progbar):
            mel = tf.transpose(mel, [0, 2, 1])  # (B, 3000, 80)
            with tf.GradientTape() as tape:
                sot = tf.fill([tf.shape(target_tokens)[0], 1], tokenizer.sot)
                decoder_input = tf.concat([sot, target_tokens[:, :-1]], axis=1)
                logits = model(mel, decoder_input, training = True)
                mask = tf.cast(tf.not_equal(target_tokens, tokenizer.special_tokens["<|pad|>"]), tf.float32)
                loss = loss_fn(target_tokens, logits, sample_weight=mask)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            progbar.set_postfix(loss=loss.numpy())

            if step % 10 == 0:
                decoded = [tokenizer.sot]
                for _ in range(100):
                    decoder_input = tf.expand_dims(tf.constant(decoded, dtype=tf.int32), axis=0)
                    pred_logits = model(mel[:1], decoder_input, True)
                    next_token = tf.argmax(pred_logits[:, -1, :], axis=-1).numpy()[0]
                    decoded.append(next_token)
                    if next_token == tokenizer.eot:
                        break
                prediction = tokenizer.decode(decoded)
                reference = tokenizer.decode(target_tokens[0].numpy().tolist())
                print(f"GT: {reference}")
                print(f"PR: {prediction}")
                print(f"WER: {wer(reference, prediction):.2f}\n")

    print("Saving model...")
    model.save_weights("trained_whisper_overfit.weights.h5")
    test_model(model, tokenizer, testing)

if __name__ == "__main__":
    main()
