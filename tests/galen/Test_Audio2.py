from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/Users/galen/git/hugginface/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
# model.cuda()

outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="/Users/galen/temp/3.wav",
    gpt_cond_len=3,
    language="en",
)
