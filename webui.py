# coding=utf-8

import os
import librosa
import base64
import io
import gradio as gr
import re
from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch
import torchaudio

from pydub import AudioSegment
from pydub.silence import detect_nonsilent, split_on_silence



from funasr import AutoModel

model = "iic/SenseVoiceSmall"
model = AutoModel(model=model,
				  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
				  vad_kwargs={"max_single_segment_time": 30000},
				  trust_remote_code=True,
				  )

import re

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
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

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def model_inference(input_wav, language, fs=16000):
	# task_abbr = {"Speech Recognition": "ASR", "Rich Text Transcription": ("ASR", "AED", "SER")}
	language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
					 "nospeech": "nospeech"}
	
	# task = "Speech Recognition" if task is None else task
	language = "auto" if len(language) < 1 else language
	selected_language = language_abbr[language]
	# selected_task = task_abbr.get(task)
	
	# print(f"input_wav: {type(input_wav)}, {input_wav[1].shape}, {input_wav}")
	
	if isinstance(input_wav, tuple):
		fs, input_wav = input_wav
		input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
		if len(input_wav.shape) > 1:
			input_wav = input_wav.mean(-1)
		if fs != 16000:
			print(f"audio_fs: {fs}")
			resampler = torchaudio.transforms.Resample(fs, 16000)
			input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
			input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
	
	
	merge_vad = True #False if selected_task == "ASR" else True
	print(f"language: {language}, merge_vad: {merge_vad}")
	text = model.generate(input=input_wav,
						  cache={},
						  language=language,
						  use_itn=True,
						  batch_size_s=0, merge_vad=merge_vad)
	
	print(text)
	text = text[0]["text"]
	text = format_str_v3(text)
	
	print(text)
	
	return text


def model_inference(input_wav, language, fs=16000):
    language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
                     "nospeech": "nospeech"}
    
    language = "auto" if len(language) < 1 else language
    selected_language = language_abbr[language]
    
    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            print(f"audio_fs: {fs}")
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
    
    merge_vad = True
    print(f"language: {language}, merge_vad: {merge_vad}")
    text = model.generate(input=input_wav,
                          cache={},
                          language=language,
                          use_itn=True,
                          batch_size_s=0, merge_vad=merge_vad)
    
    print(text)
    text = text[0]["text"]
    text = format_str_v3(text)
    
    print(text)
    
    return text
# 转换毫秒到SRT时间格式
def ms_to_srt_time(ms):
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

def segment_and_transcribe(input_wav, language, output_path = "outsrt/transcription.srt"):

    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
        input_wav = (input_wav * np.iinfo(np.int16).max).astype(np.int16)

    audio_segment = AudioSegment(
        input_wav.tobytes(), 
        frame_rate=16000, 
        sample_width=input_wav.dtype.itemsize, 
        channels=1
    )

    nonsilent_ranges = detect_nonsilent(audio_segment, min_silence_len=500, silence_thresh=-40)
    srt_content = []
    start_time = 0
    counter = 1

    for start, end in nonsilent_ranges:
        # 处理静音片段
        if start_time < start:
            srt_content.append(f"{counter}\n{ms_to_srt_time(start_time)} --> {ms_to_srt_time(start)}\n\n")
            counter += 1
        # 处理非静音片段
        chunk = audio_segment[start:end]
        segment_wav = np.array(chunk.get_array_of_samples())
        segment_wav = (segment_wav / np.iinfo(np.int16).max).astype(np.float32)
        
        transcription = model_inference(segment_wav, language)
        
        srt_content.append(f"{counter}\n{ms_to_srt_time(start)} --> {ms_to_srt_time(end)}\n{transcription}\n")
        counter += 1
        start_time = end  # 更新下一个片段的开始时间
    
    # 处理音频末尾的静音段
    if start_time < len(audio_segment):
        srt_content.append(f"{counter}\n{ms_to_srt_time(start_time)} --> {ms_to_srt_time(len(audio_segment))}\n\n")

    srt_text = "\n".join(srt_content)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    
    print(f"SRT file saved to {output_path}")
    return srt_text


    

html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">声音理解模型：SenseVoice-Small</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small 是一个编码器基础的声音基础模型，专为快速声音理解而设计。它包含了多种特性，包括自动语音识别（ASR）、口语识别（LID）、语音情感识别（SER）和声学事件检测（AED）。SenseVoice-Small 支持中文、英文、粤语、日语和韩语的多语种识别。此外，它还提供了异常低的推理延迟，比 Whisper-small 快 7 倍，比 Whisper-large 快 17 倍。</p>
    <h2 style="font-size: 22px;margin-left: 0px;">使用方法</h2> 
    <p style="font-size: 18px;margin-left: 20px;">上传音频文件或通过麦克风输入，然后选择任务和语言。音频将被转录成相应的文本，并附带相关的情感（😊 快乐，😡 生气/兴奋，😔 悲伤）和声音事件类型（😀 笑声，🎼 音乐，👏 掌声，🤧 咳嗽和打喷嚏，😭 哭泣）。事件标签将放置在文本的前面，情感标签将放在文本的后面。</p>
    <p style="font-size: 18px;margin-left: 20px;">推荐音频输入时长不超过 30 秒。对于超过 30 秒的音频，建议本地部署。</p>
    <h2 style="font-size: 22px;margin-left: 0px;">代码仓库</h2>
    <p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice</a>: 多语种语音理解模型</p>
    <p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/modelscope/FunASR" target="_blank">FunASR</a>: 基础语音识别工具包</p>
    <p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">CosyVoice</a>: 高品质多语种TTS模型</p>
</div>
"""


def launch():
	with gr.Blocks(theme=gr.themes.Soft()) as demo:
		# gr.Markdown(description)
		gr.HTML(html_content)
		with gr.Row():
			with gr.Column():
				audio_inputs = gr.Audio(label="上传音频")
				with gr.Accordion("Configuration"):
					# task_inputs = gr.Radio(choices=["Speech Recognition", "Rich Text Transcription"],
					# 					   value="Speech Recognition", label="Task")
					language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
												  value="auto",
												  label="Language")
			with gr.Column():		
				fn_button = gr.Button("一般输出", variant="primary")
				fn_button1 = gr.Button("字幕输出", variant="primary")				
				text_outputs = gr.Textbox(label="输出结果")
		
		fn_button.click(model_inference, inputs=[audio_inputs, language_inputs], outputs=text_outputs)
		fn_button1.click(segment_and_transcribe, inputs=[audio_inputs, language_inputs], outputs=text_outputs)
		# with gr.Accordion("More examples"):
		# 	gr.HTML(centered_table_html)
	demo.launch()


if __name__ == "__main__":
	# iface.launch()
	launch()


