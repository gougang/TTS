import argparse
import os
import sys
import tempfile

import torch
import torchaudio
import requests
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from flask import Flask

# 创建一个Flask应用
app = Flask(__name__)


# 定义一个路由，当访问根目录时返回"Hello, World!"消息
@app.route('/jupyter-port/1121271/8808/hello')
def hello_world():
    return 'Hello, World!'


# export PYTHONPATH=/root/github/TTS:$PYTHONPATH
def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


XTTS_MODEL = None


def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, speaker_file_path='/root/model/XTTS-v2/speakers_xtts.pth',
                               checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "Model Loaded!"


def run_tts(lang, tts_text, speaker_audio_file):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file,
                                                                             gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
                                                                             max_ref_length=XTTS_MODEL.config.max_ref_len,
                                                                             sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    out = XTTS_MODEL.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=XTTS_MODEL.config.temperature,  # Add custom parameters here
        length_penalty=XTTS_MODEL.config.length_penalty,
        repetition_penalty=XTTS_MODEL.config.repetition_penalty,
        top_k=XTTS_MODEL.config.top_k,
        top_p=XTTS_MODEL.config.top_p,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file


# define a logger to redirect
class Logger:
    def __init__(self, filename="log.out"):
        self.log_file = filename
        self.terminal = sys.stdout
        self.log = open(self.log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


# redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def uploadWav(file_path):
    url = 'http://127.0.0.1:7860/ai-human-video/upload'  # 服务器的上传API URL

    # 打开文件并上传
    with open(file_path, 'rb') as file:
        files = {'file': (file.name, file, 'multipart/form-data')}
        response = requests.post(url, files=files)

    print('exec-result-start', response.text, 'exec-result-end')  # 打印服务器响应


@app.route('/jupyter-port/1121271/8808/text-to-speach')
def create(style, text, reference):
    xtts_checkpoint = "/home/hotel/xtts/models/checkpoint/gaoling/model.pth"
    xtts_config = "/home/hotel/xtts/models/checkpoint/gaoling/config.json"
    xtts_vocab = "/home/hotel/xtts/models/checkpoint/gaoling/vocab.json"

    if style == 'ch':
        xtts_checkpoint = "/home/hotel/xtts/models/checkpoint/chenghong/checkpoint_2000.pth"
        xtts_config = "/home/hotel/xtts/models/checkpoint/chenghong/config.json"
        xtts_vocab = "/home/hotel/xtts/models/checkpoint/chenghong/vocab.json"


    tts_language = "zh"
    tts_text = text

    speaker_reference_audio = "/home/hotel/xtts/models/speaker/lishuai.wav"

    if tts_text is None or len(tts_text) <= 0 or len(tts_text) > 50:
        print("请输入不多于50个字符")
    else:

        load_model(xtts_checkpoint, xtts_config, xtts_vocab)
        [progress_gen, tts_output_audio, reference_audio] = run_tts(tts_language, tts_text,
                                                                    speaker_reference_audio)

        # 上传文件
        uploadWav(tts_output_audio)
        print("progress_gen", progress_gen)
        print("tts_output_audio", tts_output_audio)
        print("reference_audio", reference_audio)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8808)
