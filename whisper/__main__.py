import os
import random
import tensorflow as tf
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import Tokenizer
from whisper.audio import process_all_audio_files, load_audio, log_mel_spectrogram, pad_or_trim
from whisper.decoder import decode, DecodingOptions
from whisper.tokenizer import get_tokenizer

def load_dataset(data_dir, tokenizer, max_audio_len=3000):
    """
    Load dataset from LibriSpeech-like directory structure.
    Each audio file has a corresponding .txt file.
    """
    dataset = process_all_audio_files(data_dir)

    def gen():
        for mel, tokens in dataset:
            yield mel, tokens

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(80, max_audio_len), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    )

def split_dev_clean(dev_dir, split_ratio=0.8):
    """
    Split LibriSpeech dev-clean into training and validation sets.
    """
    all_samples = []
    for root, _, files in os.walk(dev_dir):
        for directory in files:
            if directory.endswith(".flac") or directory.endswith(".wav"):
                audio_path = os.path.join(root, directory)
                text_path = audio_path.rsplit(".", 1)[0] + ".txt"
                # if os.path.exists(text_path):
                all_samples.append((audio_path, text_path))

    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * split_ratio)
    print(all_samples)
    return all_samples[:split_idx], all_samples[split_idx:]

def train(model: Whisper, tokenizer: Tokenizer, dataset: tf.data.Dataset, epochs: int = 5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (mel, target_tokens) in enumerate(dataset):
            with tf.GradientTape() as tape:
                audio_features = model.encoder(tf.expand_dims(tf.transpose(mel), axis=0))
                input_tokens = tf.expand_dims(tf.constant([tokenizer.sot] + target_tokens[:-1]), axis=0)
                target_tokens = tf.expand_dims(target_tokens, axis=0)
                logits = model.decoder(input_tokens, audio_features)
                loss = loss_fn(target_tokens, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.numpy():.4f}")

def test(model: Whisper, tokenizer: Tokenizer, test_dir: str):
    print("Testing on directory:", test_dir)
    for root, _, files in os.walk(test_dir):
        for file in files:
            if not file.endswith(".wav") and not file.endswith(".flac"):
                continue
            path = os.path.join(root, file)
            audio = load_audio(path)
            mel = log_mel_spectrogram(pad_or_trim(audio))
            mel_tensor = tf.expand_dims(mel, axis=0)
            mel_tensor = tf.transpose(mel_tensor, [0, 2, 1])

            options = DecodingOptions()
            result = decode(model.encoder, model.decoder, tokenizer, mel_tensor, options)
            print(f"{file}: {result.text}")

def main():
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=3000,
        n_audio_state=384,
        n_audio_head=6,
        n_audio_layer=4,
        n_vocab=51864,
        n_text_ctx=448,
        n_text_state=384,
        n_text_head=6,
        n_text_layer=4
    )
    model = Whisper(dims)

    tokenizer = get_tokenizer(multilingual=False)


    print("Splitting dev-clean dataset...")
    train_split, val_split = split_dev_clean("/Users/robertogonzales/Desktop/DL/WhisperData/librispeech/dev-clean")

    print("Loading training data from dev-clean split...")
    train_dataset = []
    for audio_path, text_path in train_split:
        with open(text_path, 'r') as f:
            transcription = f.read().strip()
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(pad_or_trim(audio))
        tokens = tokenizer.encode(transcription)
        train_dataset.append((mel, tokens))

    def gen():
        for mel, tokens in train_dataset:
            yield mel, tokens

    train_dataset_tf = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(80, 3000), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    ).padded_batch(1)

    train(model, tokenizer, train_dataset_tf, epochs=5)

    print("Running evaluation on validation split...")
    test(model, tokenizer, val_split)
    # for audio_path, text_path in val_split:
    #     with open(text_path, 'r') as f:
    #         reference = f.read().strip()
    #     audio = load_audio(audio_path)
    #     mel = log_mel_spectrogram(pad_or_trim(audio))
    #     mel_tensor = tf.expand_dims(mel, axis=0)
    #     mel_tensor = tf.transpose(mel_tensor, [0, 2, 1])

    #     options = DecodingOptions()
    #     result = decode(model.encoder, model.decoder, tokenizer, mel_tensor, options)
    #     print(f"Predicted: {result.text}")
    #     print(f"Reference: {reference}\n")

if __name__ == "__main__":
    main()
