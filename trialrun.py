import whisper
import whisper.__main__ as __main__

model = whisper.load_model("tiny")
result = model.transcribe("audio\8842-302196-0000.flac")
print(result["text"])


# import whisper
# import decoder

# model = whisper.load_model("tiny")

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("audio\8842-302196-0000.flac")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels) #.to(model.device)

# # detect the spoken language
# # _, probs = model.detect_language(mel)
# # print(f"Detected language: {max(probs, key=probs.get)}") ## tiny cannot detect  

# # decode the audio
# options = whisper.DecodingOptions()
# result = decoder(model, mel, options)

# # print the recognized text
# print(result.text)

