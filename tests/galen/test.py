import torch
from TTS.api import TTS
import requests
import argparse


def uploadWav(file_path):
    no_proxy = {
        "http": None,
        "https": None,
    }
    url = 'http://127.0.0.1:7001/ai-human-video/upload'  # 服务器的上传API URL

    # 打开文件并上传
    with open(file_path, 'rb') as file:
        files = {'file': (file.name, file, 'multipart/form-data')}
        response = requests.post(url, files=files, proxies=no_proxy)

    print('exec result: start',response.text,'exec result: start')  # 打印服务器响应


def ensure_ends_with_slash(url):
    if url == "":
        return ""
    if not url.endswith('/'):
        return url + '/'
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="参数解析")
    parser.add_argument(
        "--text",
        type=str,
        help="文本内容",
        default='',
    )
    parser.add_argument(
        "--style",
        type=str,
        help="语音类型",
        default="gl",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        help="语音类型",
        default="gl",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出路径",
        default="",
    )

    args = parser.parse_args()

    if args.text == "":
        print("text不能为空")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("use device:", device)

        base_path = "/home/hotel/xtts/models/checkpoint/gaoling"
        if "ch" == args.style:
            base_path = "/home/hotel/xtts/models/checkpoint/chenghong"

        # 生成路径
        output_path = ensure_ends_with_slash(args.output) + "output.wav"

        # 合成声音
        tts = TTS(model_path=base_path, config_path=base_path + "/config.json", progress_bar=False, gpu=False)
        tts.tts_to_file(text=args.text, language="zh-cn",
                        speaker_wav="/home/hotel/xtts/models/checkpoint/gaoling/gaoling_00053.wav",
                        file_path=output_path)
        tts.to(device)

        # 上传文件
        uploadWav(output_path)