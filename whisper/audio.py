import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
import librosa

import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE 
# N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2 
# FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH) 
# TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN) 

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
        
    window = tf.signal.hann_window(N_FFT)
    stft = tf.signal.stft(audio, frame_length=N_FFT, frame_step=HOP_LENGTH, fft_length=N_FFT, window_fn=lambda _:window, pad_end=False)
    magnitudes = tf.abs(stft[..., :-1]) ** 2
    
    filters = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=n_mels,  num_spectrogram_bins=N_FFT // 2, sample_rate=16000, lower_edge_hertz=0.0,  upper_edge_hertz=8000.0, dtype=tf.float32)
    
    mel_spec = tf.matmul(magnitudes, filters)

    log_spec = tf.math.log(tf.clip_by_value(mel_spec, 1e-10, tf.reduce_max(mel_spec)))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - 8.0)

    log_spec = (log_spec + 4.0) / 4.0

    log_mel = tf.transpose(log_spec)

    return log_mel









