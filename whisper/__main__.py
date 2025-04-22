import os
import random
import numpy as np
import tensorflow as tf
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import Tokenizer
from whisper.audio import process_all_audio_files, load_audio, log_mel_spectrogram, pad_or_trim
from whisper.decoder import decode, DecodingOptions
from whisper.tokenizer import get_tokenizer
from tqdm import tqdm

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
    Parses the centralized transcription file per folder.
    Returns a list of (audio_path, transcription_text) pairs.
    """
    all_samples = []

    for root, _, files in os.walk(dev_dir):
        trans_files = [f for f in files if f.endswith('.trans.txt')]
        for trans_file in trans_files:
            trans_path = os.path.join(root, trans_file)
            
            trans_dict = {}
            with open(trans_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utt_id, transcription = parts
                        trans_dict[utt_id] = transcription

            for utt_id in trans_dict:
                audio_file = utt_id + ".flac"
                audio_path = os.path.join(root, audio_file)
                if os.path.exists(audio_path):
                    all_samples.append((audio_path, trans_dict[utt_id]))

    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * split_ratio)
    
    return all_samples[:split_idx], all_samples[split_idx:]

def train(model: Whisper, tokenizer: Tokenizer, dataset: tf.data.Dataset, epochs: int = 5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=5*1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        if epoch == 5:
            return
        progbar = tqdm(dataset, desc="Training", unit="batch")
        for step, (mel, target_tokens) in enumerate(dataset):
            mel = tf.transpose(mel, [0, 2, 1]) 

            with tf.GradientTape() as tape:
                audio_features = model.encoder(mel) #encoded spectrogram, conv1d and sinusoidal embedding

                sot = tf.fill([tf.shape(target_tokens)[0], 1], tokenizer.sot) 
                decoder_input = tf.concat([sot, target_tokens[:, :-1]], axis=1)  

                logits = model.decoder(decoder_input, audio_features)  
                loss = loss_fn(target_tokens, logits)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            for v, g in zip(model.trainable_variables, gradients):
                if g is None:
                    print(f"No gradient for: {v.name}")
                # else:
                #     print(f"Gradient OK for: {v.name}")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            progbar.set_postfix(loss=loss.numpy())

            if step % 10 == 0:
                # print(f"Step {step}: Loss = {loss.numpy():.4f}")
                tqdm.write(f"Step {step}: Loss = {loss:.4f}")


def main():
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=3000,
        n_audio_state=384,
        n_audio_head=6,
        n_audio_layer=4,
        n_vocab=100262,
        n_text_ctx=448,
        n_text_state=384,
        n_text_head=6,
        n_text_layer=4
    )
    model = Whisper(dims)
    tokenizer = get_tokenizer()


    print("Splitting dev-clean dataset...")
    train_split, val_split = split_dev_clean("C:/Users/anant/Desktop/whisperdata/sample/121123")
    print("Loading training data from dev-clean split...")
    train_dataset = []
    for audio_path, transcription in train_split:
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(pad_or_trim(audio))
        tokens = tokenizer.encode(transcription)
        train_dataset.append((mel, tokens))

    def gen():
        for mel, tokens in train_dataset:
            mel_fixed = pad_or_trim(mel, length=3000, axis=-1)
            yield mel_fixed, tokens

    train_dataset_tf = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(80, 3000), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    ).padded_batch(1)

    print("Starting training...")
    train(model, tokenizer, train_dataset_tf, epochs=5)
    dummy_mel = tf.zeros((1, 3000, 80))
    dummy_tokens = tf.zeros((1, 10), dtype=tf.int32)
    _ = model(dummy_mel, dummy_tokens)
    model.save_weights("trained_whisper_model.weights.h5")

    print("Running evaluation on validation split...")
    print("Decode sot: ", tokenizer.decode([tokenizer.sot]))
    for audio_path, transcription in val_split:
        print("audio: ", audio_path, "transcription: ", transcription)
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(pad_or_trim(audio, length=3000, axis=-1))
        mel_tensor = tf.expand_dims(mel, axis=0)  # (1, 80, 3000)
        mel_tensor = tf.transpose(mel_tensor, [0, 2, 1])  # (1, 3000, 80) for encoder

        options = DecodingOptions()

        result = decode(model.encoder, model.decoder, tokenizer, mel_tensor, options)
        first_token = result.tokens.numpy()[0]
        print("First token ID:", first_token)
        print("First token decoded:", tokenizer.decode([first_token]))
        print(f"Predicted: {result.text}\n")
        print(f"WER: {wer_func(transcription, result.text)}")

    model.save_weights("trained_whisper_model.weights.h5")

def wer_func(ref, hyp):
    ref = ref.split()
    hyp = hyp.split()
    len_r = len(ref)
    len_h = len(hyp)

    mat = np.zeros((len_r+1,len_h+1))

    for i in range(len_r+1):
        mat[i][0] = i
    for j in range(len_h+1):
        mat[0][j] = j

    for i in range(1,len_r+1):
        for j in range(1, len_h+1):
            if ref[i-1]==hyp[j-1]:
                mat[i][j] = mat[i-1][j-1]
            else:
                sub = mat[i-1, j-1] + 1
                ins = mat[i,j-1] + 1
                del_w = mat[i-1,j] + 1
                mat[i,j] = min(sub,ins,del_w)
    
    return mat[len_r,len_h]/len_r



if __name__ == "__main__":
    main()