import whisper.__main__ as __main__

model = __main__.load_model("tiny")
result = model.transcribe("audio/8842-302196-0000.flac")
print(result["text"])