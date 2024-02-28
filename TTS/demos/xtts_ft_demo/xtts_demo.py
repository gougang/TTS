import argparse
import os
import sys
import tempfile

import gradio as gr
import librosa.display
import numpy as np
from datetime import datetime

import os
import torch
import torchaudio
import traceback
from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


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


def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()


if __name__ == "__main__":

    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化日期时间为字符串
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
                    """
                    Example runs:
                    python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port 
                    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio demo. Default: 5003",
        default=5003,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved) Default: /tmp/xtts_ft/",
        default="/home/hotel/xtts/training-result/" + formatted_datetime,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 10",
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 4",
        default=4,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=11,
    )

    args = parser.parse_args()

    with gr.Blocks() as demo:
        with gr.Tab("1 - Data processing"):
            out_path = gr.Textbox(
                label="Output path (where data and checkpoints will be saved):",
                value=args.out_path,
            )
            # upload_file = gr.Audio(
            #     sources="upload",
            #     label="Select here the audio files that you want to use for XTTS trainining !",
            #     type="filepath",
            # )
            upload_file = gr.File(
                file_count="multiple",
                label="Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)",
            )
            lang = gr.Dropdown(
                label="Dataset Language",
                value="zh",
                choices=[
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "ru",
                    "nl",
                    "cs",
                    "ar",
                    "zh",
                    "hu",
                    "ko",
                    "ja"
                ],
            )
            progress_data = gr.Label(
                label="Progress:"
            )
            logs = gr.Textbox(
                label="Logs:",
                interactive=False,
            )
            demo.load(read_logs, None, logs, every=1)

            prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")


            def preprocess_dataset(audio_path, language, out_path, progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
                out_path = os.path.join(out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
                if audio_path is None:
                    audio_path = [
                        "/home/hotel/xtts/training-data/gl/10_33_gaoling_00032_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/11_32_gaoling_00031_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/12_31_gaoling_00030_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/13_30_gaoling_00029_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/14_29_gaoling_00028_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/1_42_gaoling_00041_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/15_28_gaoling_00027_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/16_27_gaoling_00026_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/17_26_gaoling_00025_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/18_25_gaoling_00024_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/19_24_gaoling_00023_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/20_23_gaoling_00022_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/21_22_gaoling_00021_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/22_21_gaoling_00020_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/23_20_gaoling_00019_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/24_19_gaoling_00018_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/2_41_gaoling_00040_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/25_18_gaoling_00017_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/26_17_gaoling_00016_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/27_16_gaoling_00015_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/28_15_gaoling_00014_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/29_14_gaoling_00013_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/30_13_gaoling_00012_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/31_12_gaoling_00011_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/32_11_gaoling_00010_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/33_10_gaoling_00009_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/3_40_gaoling_00039_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/34_9_gaoling_00008_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/35_8_gaoling_00007_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/36_7_gaoling_00006_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/37_6_gaoling_00005_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/38_5_gaoling_00004_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/39_4_gaoling_00003_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/40_3_gaoling_00002_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/41_2_gaoling_00001_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/42_1_gaoling_00000_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/4_39_gaoling_00038_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/5_38_gaoling_00037_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/6_37_gaoling_00036_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/7_36_gaoling_00035_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/8_35_gaoling_00034_(Vocals)_(Vocals).mp3",
                        "/home/hotel/xtts/training-data/gl/9_34_gaoling_00033_(Vocals)_(Vocals).mp3"
                    ]
                if audio_path is None:
                    return "You should provide one or multiple audio files! If you provided it, probably the upload of the files is not finished yet!", "", ""
                else:
                    try:
                        train_meta, eval_meta, audio_total_size = format_audio_list(audio_path,
                                                                                    target_language=language,
                                                                                    out_path=out_path,
                                                                                    gradio_progress=progress)
                    except:
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}", "", ""

                clear_gpu_cache()

                # if audio total len is less than 2 minutes raise an error
                if audio_total_size < 120:
                    message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
                    print(message)
                    return message, "", ""

                print("Dataset Processed!")
                return "Dataset Processed!", train_meta, eval_meta

        with gr.Tab("2 - Fine-tuning XTTS Encoder"):
            train_csv = gr.Textbox(
                label="Train CSV:",
            )
            eval_csv = gr.Textbox(
                label="Eval CSV:",
            )
            num_epochs = gr.Slider(
                label="Number of epochs:",
                minimum=1,
                maximum=100,
                step=1,
                value=args.num_epochs,
            )
            batch_size = gr.Slider(
                label="Batch size:",
                minimum=2,
                maximum=512,
                step=1,
                value=args.batch_size,
            )
            grad_acumm = gr.Slider(
                label="Grad accumulation steps:",
                minimum=2,
                maximum=128,
                step=1,
                value=args.grad_acumm,
            )
            max_audio_length = gr.Slider(
                label="Max permitted audio size in seconds:",
                minimum=2,
                maximum=20,
                step=1,
                value=args.max_audio_length,
            )
            progress_train = gr.Label(
                label="Progress:"
            )
            logs_tts_train = gr.Textbox(
                label="Logs:",
                interactive=False,
            )
            demo.load(read_logs, None, logs_tts_train, every=1)
            train_btn = gr.Button(value="Step 2 - Run the training")


            def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path,
                            max_audio_length):
                clear_gpu_cache()
                if not train_csv or not eval_csv:
                    return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language,
                                                                                                         num_epochs,
                                                                                                         batch_size,
                                                                                                         grad_acumm,
                                                                                                         train_csv,
                                                                                                         eval_csv,
                                                                                                         output_path=output_path,
                                                                                                         max_audio_length=max_audio_length)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""

                # copy original files to avoid parameters changes issues
                os.system(f"cp {config_path} {exp_path}")
                os.system(f"cp {vocab_file} {exp_path}")

                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                print("Model training done!")
                clear_gpu_cache()
                return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav

        with gr.Tab("3 - Inference"):
            with gr.Row():
                with gr.Column() as col1:
                    xtts_checkpoint = gr.Textbox(
                        label="XTTS checkpoint path:",
                        value="",
                    )
                    xtts_config = gr.Textbox(
                        label="XTTS config path:",
                        value="",
                    )

                    xtts_vocab = gr.Textbox(
                        label="XTTS vocab path:",
                        value="",
                    )
                    progress_load = gr.Label(
                        label="Progress:"
                    )
                    load_btn = gr.Button(value="Step 3 - Load Fine-tuned XTTS model")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(
                        label="Speaker reference audio:",
                        value="",
                    )
                    tts_language = gr.Dropdown(
                        label="Language",
                        value="zh",
                        choices=[
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "zh",
                            "hu",
                            "ko",
                            "ja",
                        ]
                    )
                    tts_text = gr.Textbox(
                        label="Input Text.",
                        value="This model sounds really good and above all, it's reasonably fast.",
                    )
                    tts_btn = gr.Button(value="Step 4 - Inference")

                with gr.Column() as col3:
                    progress_gen = gr.Label(
                        label="Progress:"
                    )
                    tts_output_audio = gr.Audio(label="Generated Audio.")
                    reference_audio = gr.Audio(label="Reference audio used.")

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    upload_file,
                    lang,
                    out_path,
                ],
                outputs=[
                    progress_data,
                    train_csv,
                    eval_csv,
                ],
            )

            train_btn.click(
                fn=train_model,
                inputs=[
                    lang,
                    train_csv,
                    eval_csv,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    out_path,
                    max_audio_length,
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint, speaker_reference_audio],
            )

            load_btn.click(
                fn=load_model,
                inputs=[
                    xtts_checkpoint,
                    xtts_config,
                    xtts_vocab
                ],
                outputs=[progress_load],
            )

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                ],
                outputs=[progress_gen, tts_output_audio, reference_audio],
            )

    demo.launch(
        share=True,
        debug=False,
        server_port=8808,
        root_path="/jupyter-port/1121271/8808/xtts_ft_demo",
        server_name="0.0.0.0"
    )
