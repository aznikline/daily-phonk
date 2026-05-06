import os
import datetime as _dt
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Literal, Optional

import numpy as np
import soundfile as sf
import yaml

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

logger = logging.getLogger("daily-phonk.generator")


@dataclass
class StyleConfig:
    id: str
    probability: float
    prompt: str


@dataclass
class OutputConfig:
    root_dir: str
    filename_prefix: str


@dataclass
class AppConfig:
    backend: Literal["neural", "dsp"]
    duration_seconds: int
    sample_rate: int
    bit_depth: int
    styles: List[StyleConfig]
    output: OutputConfig
    neural: Dict[str, Any]


def _load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    styles = [StyleConfig(**s) for s in raw["styles"]]
    output_cfg = OutputConfig(**raw["output"])

    return AppConfig(
        backend=raw.get("backend", "neural"),
        duration_seconds=int(raw["duration_seconds"]),
        sample_rate=int(raw["sample_rate"]),
        bit_depth=int(raw.get("bit_depth", 32)),
        styles=styles,
        output=output_cfg,
        neural=raw.get("neural", {}),
    )


def _choose_style(styles: List[StyleConfig]) -> StyleConfig:
    probs = np.array([s.probability for s in styles], dtype=float)
    total = probs.sum()
    if total <= 0:
        # fallback: uniform
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
    idx = np.random.choice(len(styles), p=probs)
    return styles[idx]


def _ensure_output_dir(root_dir: str, day: _dt.date) -> str:
    year = f"{day.year:04d}"
    month = f"{day.month:02d}"
    day_s = f"{day.day:02d}"
    path = os.path.join(root_dir, year, month, day_s)
    os.makedirs(path, exist_ok=True)
    return path


def _build_filename(prefix: str, style_id: str, day: _dt.date) -> str:
    date_str = day.strftime("%Y%m%d")
    return f"{date_str}_{style_id}_{prefix}.wav"


def _save_audio(
    audio: np.ndarray,
    sample_rate: int,
    path: str,
    bit_depth: int = 32,
) -> None:
    # 范围限制到 [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)

    subtype = "FLOAT"
    if bit_depth == 16:
        subtype = "PCM_16"
    elif bit_depth == 24:
        subtype = "PCM_24"
    elif bit_depth == 32:
        subtype = "FLOAT"

    sf.write(path, audio, samplerate=sample_rate, subtype=subtype)


# =============== 合成相关工具函数 ===============

def _envelope(length: int, attack: float, decay: float, sample_rate: int) -> np.ndarray:
    """简单 ADSR（只有 A+D），生成 0~1 的包络。"""
    t = np.arange(length) / sample_rate
    env = np.ones_like(t)
    env[t < attack] = t[t < attack] / max(attack, 1e-6)
    if decay > 0:
        d_mask = t >= attack
        env[d_mask] *= np.exp(-(t[d_mask] - attack) / decay)
    return env


def _phonk_bass(
    t: np.ndarray,
    base_freq: float,
    drive: float = 4.0,
) -> np.ndarray:
    """简单的 808 样式低音 + 失真。"""
    phase = 2 * np.pi * base_freq * t
    sig = np.sin(phase)
    # 轻微 pitch 下滑
    glide = np.exp(-t * 4.0)
    sig = np.sin(phase * glide)
    # drive
    sig = np.tanh(sig * drive)
    return sig


def _cowbell(t: np.ndarray, freq1: float, freq2: float, sample_rate: int) -> np.ndarray:
    """简易 cowbell 声音（两个三角波叠加 + 窄包络）。"""
    tri1 = 2 * np.abs(2 * ((freq1 * t) % 1.0) - 1) - 1
    tri2 = 2 * np.abs(2 * ((freq2 * t) % 1.0) - 1) - 1
    sig = (tri1 + tri2) * 0.5
    env = _envelope(len(t), attack=0.001, decay=0.05, sample_rate=sample_rate)
    return sig * env


def _hat(t: np.ndarray, sample_rate: int) -> np.ndarray:
    """噪声类 hi-hat。"""
    noise = np.random.randn(len(t))
    env = _envelope(len(t), attack=0.001, decay=0.03, sample_rate=sample_rate)
    return noise * env * 0.4


def _kick(t: np.ndarray, sample_rate: int) -> np.ndarray:
    """低频 Kick。"""
    f0 = 60.0
    phase = 2 * np.pi * f0 * t * np.exp(-t * 6.0)
    sig = np.sin(phase)
    env = _envelope(len(t), attack=0.001, decay=0.2, sample_rate=sample_rate)
    return sig * env * 0.9


def _snare(t: np.ndarray, sample_rate: int) -> np.ndarray:
    noise = np.random.randn(len(t))
    env = _envelope(len(t), attack=0.001, decay=0.18, sample_rate=sample_rate)
    tone = np.sin(2 * np.pi * 200 * t) * env
    return (noise * 0.6 + tone * 0.4) * env


def _pad_voice(
    t: np.ndarray,
    freq: float,
    sample_rate: int,
    width: float = 0.008,
) -> np.ndarray:
    """柔和 pad：多相位 detune 锯齿 + 慢包络。"""
    saw = 0.0
    detunes = [-width, -width / 2, 0.0, width / 2, width]
    for d in detunes:
        saw += 2 * ((freq * (1.0 + d) * t) % 1.0) - 1.0
    saw /= len(detunes)
    env = _envelope(len(t), attack=0.2, decay=2.5, sample_rate=sample_rate)
    return saw * env * 0.4


def _lead_voice(
    t: np.ndarray,
    freq: float,
    sample_rate: int,
) -> np.ndarray:
    """lead：锐利一点的波形 + 短包络，更有旋律感。"""
    phase = 2 * np.pi * freq * t
    sig = np.sin(phase) + 0.3 * np.sin(2 * phase)
    env = _envelope(len(t), attack=0.01, decay=0.35, sample_rate=sample_rate)
    return sig * env * 0.7


def _scale_for_style(style_id: str) -> List[int]:
    """给不同风格分配一个音阶（以半音为单位，相对于根音）。"""
    if style_id.startswith("phonk") or style_id == "trap":
        # 自然小调
        return [0, 3, 5, 7, 10]
    if style_id == "dnb":
        # 多利亚调式
        return [0, 2, 3, 5, 7, 9, 10]
    # synthwave 走大调
    return [0, 4, 7, 9]


def _chord_progression(style_id: str) -> List[int]:
    """返回若干和弦度数（基于音阶索引），每个条目对应一个小节。"""
    if style_id.startswith("phonk"):
        return [0, 3, 4, 3]  # i - VI - VII - VI (在自然小调索引里近似)
    if style_id == "trap":
        return [0, 4, 3, 4]
    if style_id == "dnb":
        return [0, 2, 4, 2]
    if style_id == "synthwave":
        return [0, 2, 3, 2]  # I - III - IV - III
    return [0, 1, 2, 1]


def _note_freq(root_freq: float, semitone_offset: int) -> float:
    return root_freq * (2 ** (semitone_offset / 12.0))


def _generate_pattern_track(
    length_samples: int,
    bpm: float,
    sample_rate: int,
    style_id: str,
) -> np.ndarray:
    """根据风格合成整体节奏 + 和声 + 旋律轨道。"""
    drums = np.zeros(length_samples, dtype=np.float32)
    music = np.zeros(length_samples, dtype=np.float32)

    seconds_per_beat = 60.0 / bpm
    samples_per_beat = int(seconds_per_beat * sample_rate)
    bar_beats = 4
    bar_samples = samples_per_beat * bar_beats

    # 一些基本 pattern（kick / snare / hat）
    for start in range(0, length_samples, bar_samples):
        # kick 在 1 和 3
        for beat in (0, 2):
            s = start + beat * samples_per_beat
            e = min(s + int(0.5 * samples_per_beat), length_samples)
            t = np.arange(e - s) / sample_rate
            drums[s:e] += _kick(t, sample_rate)

        # snare 在 2 和 4
        for beat in (1, 3):
            s = start + beat * samples_per_beat
            e = min(s + int(0.5 * samples_per_beat), length_samples)
            t = np.arange(e - s) / sample_rate
            drums[s:e] += _snare(t, sample_rate)

        # hats：每 1/2 拍
        for sub in range(0, bar_beats * 2):
            s = start + int(sub * 0.5 * samples_per_beat)
            e = min(s + int(0.25 * samples_per_beat), length_samples)
            t = np.arange(e - s) / sample_rate
            hat_gain = 0.4
            if style_id in ("dnb",):
                hat_gain = 0.6
            drums[s:e] += _hat(t, sample_rate) * hat_gain

        # 特定风格的额外元素
        if style_id.startswith("phonk"):
            # 重 808 bass 在下拍
            for beat in (0, 2):
                s = start + beat * samples_per_beat
                e = min(s + samples_per_beat, length_samples)
                t = np.arange(e - s) / sample_rate
                bass = _phonk_bass(t, base_freq=43.0, drive=6.0)
                env = _envelope(len(t), attack=0.01, decay=0.6, sample_rate=sample_rate)
                music[s:e] += bass * env * 0.9

            # cowbell 在 off-beat
            for beat in (1, 3):
                s = start + beat * samples_per_beat - int(0.25 * samples_per_beat)
                if s < 0:
                    continue
                e = min(s + int(0.35 * samples_per_beat), length_samples)
                t = np.arange(e - s) / sample_rate
                cb = _cowbell(t, freq1=600.0, freq2=840.0, sample_rate=sample_rate)
                drums[s:e] += cb * 0.6

        elif style_id == "trap":
            # 更快的 hi-hat 滚奏
            for sub in range(0, bar_beats * 4):
                s = start + int(sub * 0.25 * samples_per_beat)
                e = min(s + int(0.15 * samples_per_beat), length_samples)
                t = np.arange(e - s) / sample_rate
                drums[s:e] += _hat(t, sample_rate) * 0.5

        elif style_id == "dnb":
            # break-like sub-bass
            for beat in (0, 1.5, 2.5):
                s = start + int(beat * samples_per_beat)
                e = min(s + int(0.75 * samples_per_beat), length_samples)
                t = np.arange(e - s) / sample_rate
                bass = _phonk_bass(t, base_freq=55.0, drive=3.0)
                env = _envelope(len(t), attack=0.01, decay=0.4, sample_rate=sample_rate)
                music[s:e] += bass * env * 0.7

        elif style_id == "synthwave":
            # saw bass on every beat
            for beat in range(bar_beats):
                s = start + beat * samples_per_beat
                e = min(s + samples_per_beat, length_samples)
                t = np.arange(e - s) / sample_rate
                saw = 2 * ((80.0 * t) % 1.0) - 1.0
                env = _envelope(len(t), attack=0.02, decay=0.8, sample_rate=sample_rate)
                music[s:e] += saw * env * 0.4

        # ===== 和声：pad 和旋律 =====
        scale = _scale_for_style(style_id)
        chords = _chord_progression(style_id)
        # 根音：phonk/trap 用低一点，其他高一点
        if style_id.startswith("phonk") or style_id == "trap":
            root_freq = 55.0  # A1
        else:
            root_freq = 82.41  # E2

        bar_index = (start // bar_samples) % len(chords)
        chord_degree = chords[bar_index] % len(scale)

        # 构建三和弦（root, third, fifth）
        chord_semitones = [
            scale[chord_degree],
            scale[(chord_degree + 2) % len(scale)],
            scale[(chord_degree + 4) % len(scale)],
        ]

        # pad 覆盖整小节
        ps = start
        pe = min(start + bar_samples * 2, length_samples)  # pad 稍微跨两小节更连贯
        pt = np.arange(pe - ps) / sample_rate
        pad_sig = np.zeros_like(pt)
        for semi in chord_semitones:
            pad_sig += _pad_voice(pt, _note_freq(root_freq, semi), sample_rate)
        pad_sig /= max(len(chord_semitones), 1)
        music[ps:pe] += pad_sig[: pe - ps] * (0.4 if "chill" in style_id or style_id == "synthwave" else 0.3)

        # lead：在每小节的 2、3 拍附近随机几个音
        notes_per_bar = 4 if style_id in ("dnb", "trap") else 3
        for n in range(notes_per_bar):
            beat_pos = 1.0 + n * (2.0 / notes_per_bar)  # 从第二拍开始排
            s = start + int(beat_pos * samples_per_beat)
            e = min(s + int(0.75 * samples_per_beat), length_samples)
            if e <= s:
                continue
            lt = np.arange(e - s) / sample_rate
            # 在音阶里随机抓一个音，高一点当旋律
            degree = np.random.choice(len(scale))
            semi = scale[degree] + (12 if np.random.rand() < 0.7 else 0)
            freq = _note_freq(root_freq, semi)
            lead = _lead_voice(lt, freq, sample_rate)
            gain = 0.25 if style_id.startswith("phonk") else 0.2
            music[s:e] += lead * gain

    # 鼓和音乐混合，再做轻微压缩 / 限幅
    mix = drums * 0.9 + music
    mix = np.tanh(mix * 1.5).astype(np.float32)
    return mix


class DailyPhonkGenerator:
    def __init__(self, config_path: str):
        self.config = _load_config(config_path)
        self._neural_model: Optional[MusicgenForConditionalGeneration] = None
        self._neural_processor: Optional[AutoProcessor] = None

    def generate_custom(self, prompt: str, style_id: str, duration: int = None, backend: str = None, seed: int | None = None) -> Dict[str, Any]:
        """
        通过自定义提示词和风格生成曲子（用于 Web API）。
        """
        if seed is not None:
            np.random.seed(seed)
            
        today = _dt.date.today()
        style = StyleConfig(id=style_id, probability=1.0, prompt=prompt)
        
        target_backend = backend or self.config.backend
        old_duration = self.config.duration_seconds
        if duration:
            self.config.duration_seconds = duration

        logger.info(f"自定义生成风格: {style.id}, 后端: {target_backend}")

        if target_backend == "neural":
            try:
                audio = self._generate_neural(style)
            except Exception as e:
                logger.error(f"神经网络生成失败: {e}，回退使用 DSP 合成")
                audio = self._generate_dsp(style)
        else:
            audio = self._generate_dsp(style)
            
        # 恢复配置时长
        if duration:
            self.config.duration_seconds = old_duration

        out_dir = _ensure_output_dir(self.config.output.root_dir, today)
        filename = _build_filename(self.config.output.filename_prefix, style.id, today)
        # 为防止覆盖，加上时间戳
        ts = _dt.datetime.now().strftime("%H%M%S")
        filename = filename.replace(".wav", f"_{ts}.wav")
        out_path = os.path.join(out_dir, filename)

        _save_audio(
            audio=audio,
            sample_rate=self.config.sample_rate,
            path=out_path,
            bit_depth=self.config.bit_depth,
        )

        logger.info(f"已保存: {out_path}")
        return {
            "path": out_path,
            "style_id": style.id,
            "style_prompt": style.prompt,
            "date": today.strftime("%Y-%m-%d"),
            "backend": target_backend,
            "duration": duration or self.config.duration_seconds
        }

    def generate_once(self, seed: int | None = None) -> Dict[str, Any]:
        """
        生成一首曲子，返回生成的字典信息。
        """
        if seed is not None:
            np.random.seed(seed)

        today = _dt.date.today()
        style = _choose_style(self.config.styles)

        logger.info(f"选择风格: {style.id}")

        if self.config.backend == "neural":
            try:
                audio = self._generate_neural(style)
            except Exception as e:
                logger.error(f"神经网络生成失败: {e}，回退使用 DSP 合成")
                audio = self._generate_dsp(style)
        else:
            audio = self._generate_dsp(style)

        out_dir = _ensure_output_dir(self.config.output.root_dir, today)
        filename = _build_filename(self.config.output.filename_prefix, style.id, today)
        out_path = os.path.join(out_dir, filename)

        _save_audio(
            audio=audio,
            sample_rate=self.config.sample_rate,
            path=out_path,
            bit_depth=self.config.bit_depth,
        )

        logger.info(f"已保存: {out_path}")
        return {
            "path": out_path,
            "style_id": style.id,
            "style_prompt": style.prompt,
            "date": today.strftime("%Y-%m-%d"),
            "backend": self.config.backend,
            "duration": self.config.duration_seconds
        }

    # ===== DSP 后端 =====

    def _generate_dsp(self, style: StyleConfig) -> np.ndarray:
        # 不同风格使用不同比 BPM
        if style.id.startswith("phonk"):
            bpm = 90.0
        elif style.id == "trap":
            bpm = 140.0
        elif style.id == "dnb":
            bpm = 172.0
        elif style.id == "synthwave":
            bpm = 100.0
        else:
            bpm = 110.0

        total_samples = int(self.config.duration_seconds * self.config.sample_rate)
        mono = _generate_pattern_track(
            length_samples=total_samples,
            bpm=bpm,
            sample_rate=self.config.sample_rate,
            style_id=style.id,
        )

        # 简单做一点立体声：复制并加一点不同噪声/延迟
        delay = int(0.002 * self.config.sample_rate)  # 2ms
        stereo = np.zeros((total_samples, 2), dtype=np.float32)
        stereo[:, 0] = mono
        stereo[delay:, 1] = mono[:-delay]
        stereo[:, 1] += (np.random.randn(total_samples) * 0.01).astype(np.float32)
        return stereo

    # ===== 神经网络后端（MusicGen via transformers）=====

    def _lazy_load_neural(self) -> None:
        if self._neural_model is not None and self._neural_processor is not None:
            return

        model_name = self.config.neural.get("model_name", "facebook/musicgen-medium")
        device = self.config.neural.get("device", "cpu")

        logger.info(f"加载模型: {model_name} ({device})")
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        self._neural_processor = processor
        self._neural_model = model

    def _generate_neural(self, style: StyleConfig) -> np.ndarray:
        self._lazy_load_neural()
        assert self._neural_model is not None
        assert self._neural_processor is not None

        cfg = self.config.neural
        device = cfg.get("device", "cpu")
        guidance_scale = float(cfg.get("guidance_scale", 3.0))
        max_new_tokens = int(cfg.get("max_new_tokens", 1024))

        prompt = style.prompt
        logger.info(f"提示词: {prompt}")

        inputs = self._neural_processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            audio_values = self._neural_model.generate(
                **inputs,
                guidance_scale=guidance_scale,
                max_new_tokens=max_new_tokens,
            )

        # transformers' MusicgenForConditionalGeneration returns waveform directly
        # shape is usually (batch_size, num_channels, sequence_length)
        audio = audio_values[0].cpu().numpy()  # (channels, samples)

        # 转为 (samples, channels)
        if audio.ndim == 2:
            audio = audio.T

        # 如果采样率和目标不一致，让 soundfile 重采样责任交给播放器（大多数播放器兼容）
        return audio.astype(np.float32)


__all__ = [
    "DailyPhonkGenerator",
]


