**\# Title:** Whisper

**\# Who:** Julie Chung (hchung33), Alison Zeng (avzeng), Anant Saraf (asaraf8), Roberto Gonzales Matos (rgonza27)

**\# Introduction:** What problem are you trying to solve and why?

In the original paper, OpenAI’s Whisper attempts to create a model that can predict large amounts of transcripts of audio on the internet. This paper focuses on producing accurate results that reach a similar performance as supervised learning in a zeroshot transfer setting, reducing the need for fine tuning. The original code utilizes Pytorch, so for this project, we aimed to understand and re-implemented the project using Tensorflow with less data. This project is a structured prediction, in which input audio sequence produces a sequence of tokens, which is the transcribed text as structured output. 

**\# Methodology:** What is the architecture of your model?

Whisper is an encoder-decoder Transformer sequence-to-sequence model. We convert audio files into spectrograms, which are then passed to an encoder which processes the input audio features. Then a decoder takes the encoded output and previous tokens to predict the next token. Our model uses 4 encoder blocks and 4 decoder blocks, each with 6 attention heads. We used teacher forcing in our training, so for each token the model is given the ground truth token rather than using its own previous prediction. We trained the model on the LibriSpeech ASR corpus, building a standard training loop with GradientTape to update the weights for a desired number of epochs over batches of the training dataset.

**\# Results**

We were able to realistically train from 50-100 epochs on the training data. As the model trained, the model learned rapidly in the first 10 epochs, with a steep decrease in our loss. After epoch 10, our loss continued to decrease, albeit more gradually as our model made more stable improvements. At the end of training, our loss was at 0.0586, effectively minimizing the loss. Our WER followed a similar trend, with WER being very high in the first few epochs, and then reaching a stable WER between 0.1 and 0.25. 

Whereas OpenAI’s Whisper was trained on 680,000 hours of large-scale data, we were limited to training on 5.4 hours of data due to computation and time limitations. However, we were able to successfully train our model on the dev-clean data, and after training, our model was able to accurately transcribe audio samples. We believe that training on more robust datasets such as TED-LIUM Release 3 or LibriSpeech ASR Corpus full training data (including noisy audio) would have improved the performance of the model and will continue to experiment.

**\# Challenges**

A central challenge was the overfitting of the decoder and underfitting of the encoder. Early in training, the model often prioritized memorized transcript fragments over attending to the input spectrograms, limiting its ability to generalize. In order to address this, we added a dropout layer to the decoder, as well as noise to the input tokens. Additionally, decoder attention to encoder outputs was weak, which we partially mitigated through teacher forcing to guide better alignment between input features and predictions.

Another challenge we faced was improper data preprocessing. Our conversion of audio files into spectrograms was reducing low-energy regions, which resulted in a lot of dead space within the features. To improve spectrogram quality, we refined the normalization and scaling process to preserve these regions, allowing quieter sounds to remain visible in the log-Mel spectrograms. These adjustments enhanced feature extraction and contributed to more stable model training.

Training efficiency was also hindered by the original vocabulary size. Whisper’s default \~100,000-token vocabulary increased model size, slowed training, and added unnecessary complexity. We reduced the vocabulary by nearly half through retokenization, resulting in faster convergence and improved transcription accuracy. 

Finally, the limited scale of our dataset posed a fundamental constraint. While OpenAI’s Whisper was trained on approximately 680,000 hours of diverse audio, our model was trained on only 5.4 hours of clean, scripted audiobook recordings. This several-orders-of-magnitude difference restricted model robustness and generalization to broader speech conditions.

**\# Final Reflection**

* How do you feel your project ultimately turned out? How did you do relative to your base/target/stretch goals? 

Overall, our team is satisfied with the end state of our project. We knew from the onset that the lack of data would greatly hinder our model’s potential to generalize, and it was fully expected that our model would produce inaccurate but still coherent transcriptions on data it had not seen before. 

Ultimately, the project did reach its base goal \- successfully training the model on a small subset of LibriSpeech and producing reasonable transcriptions. Our transcriptions on training were incredibly accurate and demonstrated that our model trained effectively, but was overfitting given the lack of data. Our target goal was to reach tiny.en’s WER% on LibriSpeech test-clean (5.6%), which we realized during training was unreasonable given the lack of training data and what that meant for how robust we could make our model. Our stretch goal was to run on other training and testing data to improve accuracy and subsequently upscale, but given the number of challenges we faced with regards to actually training the model, we were unable to achieve this goal. Still, during our evaluation we did see good indications that our model was performing successfully. Features such as sentence length or correct word placements for transcriptions similar to the ones in training did show up when looking at our results from our evaluation set.

* Did your model work out the way you expected it to?

Our model performed almost as expected. It overfit to the training data, and on testing data it produced smaller, higher-frequency snippets of text accurately whilst missing lower-frequency words in the transcription. Our training loss and WER were 0.05 and 0.1-0.25 respectively, which is reasonable for our dataset’s size. For reasons mentioned in the answer above, this aligned with our expectations.

* How did your approach change over time? What kind of pivots did you make, if any? Would you have done differently if you could do your project over again?

We made multiple changes over the course of the project as we learned more about the model’s behaviour:

1. We added teacher-forcing to our training mechanism, which significantly improved convergence speed and prevented high-frequency word repetition in the transcripts  
2. We remapped our tokenizer to allow for a smaller, more efficient model, improving training  
3. We adjusted our spectrogram normalization, which improved encoder input quality and led to better transcriptions  
4. Once we learnt our decoder was overfitting faster than our encoder, we added noise and dropout layers to slow decoder convergence and improve balance between both model components.

If we could have done the project again we would have better split the tasks in the project from the beginning. While at times it felt like tasks were sequential to achieve an optimal outcome there were several points where we could have sped up the process. Since we have access to OSCAR, we could have run many different iterations of our model at the same time to narrow down the errors quicker. This would have allowed us to explore more options to better generalize our results to unseen data. Areas like tokenizer and training or trying out different noise and batch size options could have been done simultaneously. Additionally, we could have started to unit test parts of our program from the start to make sure that the code we built was working properly.

* What do you think you can further improve on if you had more time?

We would, of course, train on a larger, more diverse dataset for a longer amount of time, add further optimizations to our encoder to strengthen it, and create a custom tokenizer with our own learned token representations and tweak model hyperparameters accordingly. We would also want to find a way to use the TensorFlow weights found on HuggingFace to see how it performs in comparison to ours with the same small dataset. This would give us answers to significant questions with regards to how optimal our model hyperparameters and structure are. However, this was not feasible during our implementation as our model architecture was slightly different and we could not load the weights. We originally wanted to convert the model from PyTorch to Tensorflow in individual segments at a time to ensure that the model functioned the same. However, we had to instead convert everything at once due to incompatibilities between libraries.

* What are your biggest takeaways from this project/what did you learn?

Our biggest takeaway was realizing how adapting a large, high-capacity model to a smaller dataset will result in overfitting, and the importance of tailoring the model and training process to adjust for this. Throughout the project, we were able to effectively overfit and achieve strong performance and accurate transcriptions on the original dataset. However, we lost out on a lot of the robustness and general applicability of the original OpenAI model because of this. This was even more pertinent for Whisper’s seq-to-seq, encoder-decoder structure, where we had to consider the contributions of each part of our architecture when debugging and training. We gained more experience working with seq-to-seq models and teacher forcing, and working with audio. We learnt that small changes like spectrogram normalization and improved tokenizations can make a huge difference to model results. Lastly, we learnt to work flexibly and that most of the challenges in deep learning comes from the process of optimizing training.