# Title: Whisper

# Who: Julie Chung (hchung33), Alison Zeng (avzeng), Anant Saraf (asaraf8), Roberto Gonzales Matos (rgonza27)

# Introduction: What problem are you trying to solve and why?

In the original paper, OpenAI’s Whisper attempts to create a model that can predict large amounts of transcripts of audio on the internet. This paper focuses on producing accurate results that reach a similar performance as supervised learning in a zeroshot transfer setting, reducing the need for fine tuning. The original code utilizes Pytorch, so for this project, we aimed to understand and re-implemented the project using Tensorflow. This project is a structured prediction, in which input audio sequence produces a sequence of tokens, which is the transcribed text as structured output. 

# Related Work: Are you aware of any, or is there any prior work that you drew on to do your project?

While making our own implementation of the Whisper model, we explored the paper "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" by Baevski. The document introduces a method for learning speech representations directly from raw audio using contrastive learning and a Transformer-based architecture. While Whisper and wav2vec differ in design the wav2vec 2.0 framework helped us think about the encoder design and pretraining / preprocessing strategies that could enhance low-resource performance like our implementation.

Baevski et al. (2020), wav2vec 2.0: https://arxiv.org/abs/2006.11477

# Data: What data are you using (if any)?

The current dataset we are using is LibriSpeech ASR corpus (https://www.openslr.org/12) large-scale (1000 hours) corpus of read English speech that includes clean and challenging speech, allowing training and testing to be done. Training will be done on 100 hours of English speech which is about 6.3G of data, and testing will be done on 346M of clean speech. We plan on generating spectrograms for every input audio to feed into the encoder, and use the transcript data available.

# Methodology: What is the architecture of your model?

Whisper is an encoder-decoder Transformer sequence-to-sequence model. We convert audio files into spectrograms, which are then passed to an encoder which processes the input audio features. Then a decoder takes the encoded output and previous tokens to predict the next token. We will be training the model on the LibriSpeech ASR corpus, building a standard training loop with GradientTape to update the weights for a desired number of epochs over batches of the training dataset.

One of the challenges of reimplementing Whisper in TensorFlow  is converting PyTorch-specific features. For example, TensorFlow doesn’t support key-value caching for decoder attention, which is important for model speed. We may also need to implement scaled dot product attention manually. Another challenge is simply accessing all of the data to train Whisper. The amount of data Whisper was trained on is one of its strengths, and the paper does not specifically detail the entirety of the data Whisper was trained on. While the data used to evaluate Whisper is shared, not all of these datasets are publicly available as well.


# Metrics: What constitutes “success?”

Models for automatic speech recognition are typically evaluated by Word Error Rate (WER), which evaluates the predicted transcription of an audio file with the reference transcription by calculating the ratio of substitutions, deletions, and insertions with the number of words in the reference. 

The original aim of Whisper was to conduct supervised training on a large, diverse scale, improving the generalization of a pre-trained encoder-decoder model. The goal was to create a speech recognition system that performs robustly across multiple languages and domains without requiring fine-tuning on task-specific data. This approach allowed the model to generalize well to unseen datasets, enhancing its performance across a variety of ASR benchmarks. They evaluated Whisper’s robustness by measuring the difference in expected performance between a reference dataset (LibriSpeech) and out-of-distribution datasets. When compared against 12 different datasets, Whisper achieved an average relative error reduction of 55.2% versus supervised LibriSpeech models.

Our base goal is successfully training the model on a small subset of LibriSpeech and producing reasonable transcriptions. Our target goal is to reach tiny.en’s WER% on LibriSpeech test-clean (5.6%). Our stretch goal is to 
run on other training and testing data to improve accuracy, and if that is successful, to upscale and implement one of the larger models


# Ethics: Choose 2 of the following bullet points to discuss; not all questions will be relevant to all projects so try to pick questions where there’s interesting engagement with your project. (Remember that there’s not necessarily an ethical/unethical binary; rather, we want to encourage you to think critically about your problem setup.)

What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?

The LibriSpeech ASR corpus was derived from subsets of audiobook readings from the publicly available LibriVox dataset that were ‘cleaned’ and ‘aligned’- essentially making sure that transcription aligned with the audio and that noisy audio and transcripts were not included. There are no significant concerns with regards to privacy and labor and the collection and labelling of data, respectively. However, since the dataset is primarily composed of read speech from North American English speakers, it tends to lack the variability present in conversational or accented speech. This introduces potential bias in model performance, as the system may generalize less effectively to non-American accents and spontaneous or conversational speech. We acknowledge that by using the LibriSpeech dataset, we may have well aligned data, but may be introducing bias in our system. As we integrate additional audio-transcript datasets into our training pipeline, we will carefully take into account their demographic composition—especially in terms of accent, gender, and speaking style—and make adjustments to mitigate any bias in model behavior resulting from these imbalances.

Why is Deep Learning a good approach to this problem?

Deep learning seems like the best approach to this problem because: 
Results do not need to be interpretable for transcription. This might not be the case for, say, a model that uses medical data to predict diagnoses. 
Deep learning techniques might pick up on more complex speech patterns better than traditional ML techniques 
Given its efficiency and ability to learn different speech representations, it’s incredibly useful for people with hearing disabilities and when debiased, can help different communities connect with each other



# Division of labor: Briefly outline who will be responsible for which part(s) of the project.
hchung33: Gathering data, preprocessing input audio sequence into spectrograms for training and testing
avzeng: Converting model framework (and auxiliary files) to TensorFlow
asaraf8: Collaborated with avzeng on converting model framework to Tf and fine-tuning Conv1D 
Rgonza27: Decoder, greedy and beam search approaches as well as decoding time stamps and returning the values
