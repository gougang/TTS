# from TTS.api import TTS
#
# models = TTS().list_models()
# models_name = models.list_models()[0]
# print(models_name)
# tts = TTS(models_name)
# wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
# tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "mps"
print("mps:", torch.backends.mps.is_available())
# List available 🐸TTS models
models = TTS().list_models()
models_name = models.list_models()[0]
# Init TTS
tts = TTS(model_path="/Users/galen/git/hugginface/XTTS-v2",
          config_path="/Users/galen/git/hugginface/XTTS-v2/config.json")

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(
    text="这在某些情况下可能有助于提高性能或解决一些与GPU相关的问题。但请注意，这可能会导致一些性能损失，因为CPU的计算速度通常比GPU慢!",
    speaker_wav="/Users/galen/temp/my.wav", language="zh-cn")
# Text to speech to a file
tts.tts_to_file(
    text="这在某些情况下可能有助于提高性能或解决一些与GPU相关的问题。但请注意，这可能会导致一些性能损失，因为CPU的计算速度通常比GPU慢!",
    speaker_wav="/Users/galen/temp/my.wav",
    language="zh-cn",
    file_path="/Users/galen/temp/output.wav")
