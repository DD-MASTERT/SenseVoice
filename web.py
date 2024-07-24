# 设置 FFMPEG_PATH 环境变量
import os
# ffmpeg_path = os.path.join(os.getcwd(), 'sensvoice', 'ffmpeg', 'bin')
# os.environ['FFMPEG_PATH'] = ffmpeg_path

# # 将 FFMPEG_PATH 临时添加到 PATH 环境变量中
# os.environ['PATH'] += os.pathsep + ffmpeg_path

import librosa
import base64
import io
import numpy as np
import torch
import torchaudio
from funasr import AutoModel
import sounddevice as sd
import numpy as np
import soundfile as sf
import os
import uvicorn
from fastapi import FastAPI,Request

class SenseVoiceModel:
    def __init__(self):
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=True,
        )

        self.emo_dict = {
            "<|HAPPY|>": "😊",
            "<|SAD|>": "😔",
            "<|ANGRY|>": "😡",
            "<|NEUTRAL|>": "",
            "<|FEARFUL|>": "😰",
            "<|DISGUSTED|>": "🤢",
            "<|SURPRISED|>": "😮",
        }

        self.event_dict = {
            "<|BGM|>": "🎼",
            "<|Speech|>": "",
            "<|Applause|>": "👏",
            "<|Laughter|>": "😀",
            "<|Cry|>": "😭",
            "<|Sneeze|>": "🤧",
            "<|Breath|>": "",
            "<|Cough|>": "🤧",
        }

        self.emoji_dict = {
            "<|nospeech|><|Event_UNK|>": "❓",
            "<|zh|>": "",
            "<|en|>": "",
            "<|yue|>": "",
            "<|ja|>": "",
            "<|ko|>": "",
            "<|nospeech|>": "",
            "<|HAPPY|>": "😊",
            "<|SAD|>": "😔",
            "<|ANGRY|>": "😡",
            "<|NEUTRAL|>": "",
            "<|BGM|>": "🎼",
            "<|Speech|>": "",
            "<|Applause|>": "👏",
            "<|Laughter|>": "😀",
            "<|FEARFUL|>": "😰",
            "<|DISGUSTED|>": "🤢",
            "<|SURPRISED|>": "😮",
            "<|Cry|>": "😭",
            "<|EMO_UNKNOWN|>": "",
            "<|Sneeze|>": "🤧",
            "<|Breath|>": "",
            "<|Cough|>": "😷",
            "<|Sing|>": "",
            "<|Speech_Noise|>": "",
            "<|withitn|>": "",
            "<|woitn|>": "",
            "<|GBG|>": "",
            "<|Event_UNK|>": "",
        }

        self.lang_dict = {
            "<|zh|>": "<|lang|>",
            "<|en|>": "<|lang|>",
            "<|yue|>": "<|lang|>",
            "<|ja|>": "<|lang|>",
            "<|ko|>": "<|lang|>",
            "<|nospeech|>": "<|lang|>",
        }

        self.emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
        self.event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}

    def format_str(self, s):
        for sptk in self.emoji_dict:
            s = s.replace(sptk, self.emoji_dict[sptk])
        return s

    def format_str_v2(self, s):
        sptk_dict = {}
        for sptk in self.emoji_dict:
            sptk_dict[sptk] = s.count(sptk)
            s = s.replace(sptk, "")
        emo = "<|NEUTRAL|>"
        for e in self.emo_dict:
            if sptk_dict[e] > sptk_dict[emo]:
                emo = e
        for e in self.event_dict:
            if sptk_dict[e] > 0:
                s = self.event_dict[e] + s
        s = s + self.emo_dict[emo]

        for emoji in self.emo_set.union(self.event_set):
            s = s.replace(" " + emoji, emoji)
            s = s.replace(emoji + " ", emoji)
        return s.strip()

    def format_str_v3(self, s):
        def get_emo(s):
            return s[-1] if s[-1] in self.emo_set else None

        def get_event(s):
            return s[0] if s[0] in self.event_set else None

        s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
        for lang in self.lang_dict:
            s = s.replace(lang, "<|lang|>")
        s_list = [self.format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
        new_s = " " + s_list[0]
        cur_ent_event = get_event(new_s)
        for i in range(1, len(s_list)):
            if len(s_list[i]) == 0:
                continue
            if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
                s_list[i] = s_list[i][1:]
            cur_ent_event = get_event(s_list[i])
            if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
                new_s = new_s[:-1]
            new_s += s_list[i].strip().lstrip()
        new_s = new_s.replace("The.", " ")
        return new_s.strip()

    def model_inference(self, input_wav, language, fs=16000):
        language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko", "nospeech": "nospeech"}
        language = "auto" if len(language) < 1 else language
        selected_language = language_abbr[language]

        if isinstance(input_wav, tuple):
            fs, input_wav = input_wav
            input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
            if len(input_wav.shape) > 1:
                input_wav = input_wav.mean(-1)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
                input_wav = resampler(input_wav_t[None, :])[0, :].numpy()

        merge_vad = True
        print(f"language: {language}, merge_vad: {merge_vad}")
        text = self.model.generate(input=input_wav, cache={}, language=language, use_itn=True, batch_size_s=0, merge_vad=merge_vad)
        text = text[0]["text"]
        text = self.format_str_v3(text)
        print(text)
        return text

# def record_audio_manually(output_folder, filename="audio.wav", sample_rate=44100):
#     """
#     手动控制录制的开始和结束，并保存音频数据到指定文件夹。

#     参数:
#     output_folder (str): 保存音频文件的文件夹路径。
#     filename (str): 保存音频文件的文件名（默认"recorded_audio.wav"）。
#     sample_rate (int): 音频采样率（默认44100 Hz）。

#     返回:
#     numpy.ndarray: 录制的音频数据。
#     """
#     print("按下 Enter 键开始录制，再次按下 Enter 键结束录制...")
#     input()  # 等待用户按下 Enter 键开始录制

#     audio_data = []
#     def callback(indata, frames, time, status):
#         """这是录音的回调函数"""
#         if status:
#             print(status)
#         audio_data.append(indata.copy())

#     stream = sd.InputStream(samplerate=sample_rate, channels=2, callback=callback)
#     with stream:
#         input()  # 等待用户按下 Enter 键结束录制

#     audio_data = np.concatenate(audio_data, axis=0)

#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 保存音频数据到指定文件
#     output_path = os.path.join(output_folder, filename)
#     sf.write(output_path, audio_data, sample_rate)
#     print(f"音频已保存到 {output_path}")



# Example usage
# model = SenseVoiceModel()
# input_wav="sensvoice/recorded_audios/audio.wav"
# language = "auto"
# result = model.model_inference(input_wav, language)

model = None
model = SenseVoiceModel()
app = FastAPI()

# @app.post("/open/")
# async def init_sense_voice_model():
#     global model

#     return {"message": "SenseVoiceModel initialized"}

@app.post("/run/")
async def run_model_inference():
    # 固定的输入参数
    input_wav = "recorded_audios/audio1.wav"
    language = "auto"
    
    # 执行模型推理
    result = model.model_inference(input_wav, language)
    
    # 这里没有返回值，但如果你想记录或处理结果，可以在这里进行
    # 例如，打印结果或将其存储在日志中
    # print(result)  # 假设result是可以直接打印的类型

    # 返回一个消息，表示函数已执行
    return result

# 用于关闭服务的路由
@app.post("/shutdown")
async def shutdown(request: Request):


    # 获取Uvicorn的实例
    server = request.app.state.server
    # 关闭服务器
    server.should_exit = True
    # 返回一个响应，表示服务器正在关闭
    return {"message": "Shutting down server."}

# 主函数，用于启动Uvicorn服务器
if __name__ == "__main__":
    # 将Uvicorn服务器实例存储在应用程序状态中
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=1111))
    app.state.server = server
    # 启动服务器
    server.run()

