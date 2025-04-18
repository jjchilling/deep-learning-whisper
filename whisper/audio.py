import os
import glob
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
import librosa
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf

from .utils import exact_div
from .tokenizer import get_tokenizer 


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE 
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2 
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH) 
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN) 

AUDIO_DIR = "audio/"  
EXT = "*.flac"        
MEL_SAVE_DIR = "mel/"

tokenizer = get_tokenizer(multilingual=True)
task_token = 50358  
sot_token = 50257
no_ts_token = 50362
eot_token = 50256

def load_audio(file: str, sr: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> tf.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return tf.convert_to_tensor(f[f"mel_{n_mels}"], dtype=tf.float32)

def hann_window_fn(frame_length, dtype=tf.float32):
    return tf.signal.hann_window(frame_length, dtype=dtype)

def log_mel_spectrogram(    
    audio: Union[str, np.ndarray, tf.Tensor],
    n_mels: int = 80,
    padding: int = 0,) -> tf.Tensor:

    if isinstance(audio, str):
        audio, sr = librosa.load(audio, sr=16000)
        if sr == 16000:
            audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    elif isinstance(audio, np.ndarray):
        audio = tf.convert_to_tensor(audio, tf.float32)
    elif isinstance(audio, tf.Tensor):
        audio = tf.cast(audio, dtype=tf.float32)
    else:
        print("Error has occurred, not an audio, ndarray, or tensor")

    # Unsure if device none is applicable?
    # Original: if device is not None:
    #    audio = audio.to(device)

    if padding > 0:
        audio = tf.pad(audio, [[0, padding]])
        
    window = tf.signal.hann_window(N_FFT, dtype=tf.float32)
   
    stft = tf.signal.stft(
    audio,
    frame_length=N_FFT,
    frame_step=HOP_LENGTH,
    fft_length=N_FFT,
    window_fn=hann_window_fn,
    pad_end=False
    )   

    magnitudes = tf.abs(stft) ** 2
    
    filters = mel_filters(audio.device, n_mels)
    
    mel_spec = tf.matmul(magnitudes, tf.transpose(filters))

    log_spec = tf.math.log(tf.clip_by_value(mel_spec, 1e-10, tf.reduce_max(mel_spec)))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - 8.0)

    log_spec = (log_spec + 4.0) / 4.0

    log_mel = tf.transpose(log_spec)

    return log_mel


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    if tf.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=tf.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

def save_mel(mel: tf.Tensor, original_filename: str, save_dir: str = MEL_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    mel_np = mel.numpy() if isinstance(mel, tf.Tensor) else mel
    base = os.path.basename(original_filename)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(save_dir, f"{name}_mel.npy")
    np.save(out_path, mel_np)

def process_all_audio_files(audio_dir: str = AUDIO_DIR, ext: str = EXT):
    files = sorted(glob.glob(os.path.join(audio_dir, ext)))
    mels = {}

    for file in files:
        print(f"Processing: {file}")
        try:
            audio = load_audio(file)
            audio = pad_or_trim(audio)
            mel = log_mel_spectrogram(audio)
            save_mel(mel, file, MEL_SAVE_DIR)
            mels[file] = mel
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return mels

mel = process_all_audio_files()

