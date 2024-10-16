from faster_whisper import WhisperModel

model = WhisperModel("large-v2")

segments, info = model.transcribe("/Users/galen/git/AI_Data/resources/ww/uvr/voice_1.wav")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
