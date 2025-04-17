import whisper

model = whisper.load_model("tiny")
result = model.transcribe("audio/8842-302196-0000.flac")
print(result["text"])