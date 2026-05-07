from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import mimetypes
import yaml
from typing import Optional
from pathlib import Path
import secrets

from src.generator import DailyPhonkGenerator
from src.metadata import generate_publishing_assets
from src.stats import log_generation

app = FastAPI(title="Daily Phonk Generator API")

# 配置 CORS 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    song_name: str
    style: str
    backend: Optional[str] = "neural"
    duration: Optional[int] = 30
    extra_requirements: Optional[str] = None


class PreviewRequest(BaseModel):
    song_name: str
    style: str
    backend: Optional[str] = "neural"
    duration: Optional[int] = 15
    candidates: Optional[int] = 3
    extra_requirements: Optional[str] = None


class PublishRequest(BaseModel):
    song_name: str
    style: str
    backend: Optional[str] = "neural"
    duration: Optional[int] = 45
    seed: Optional[int] = None
    extra_requirements: Optional[str] = None

# 定义不同风格的模板提示词
STYLE_TEMPLATES = {
    "dj": "high energy DJ club remix of {song_name}, heavy EDM bass, fast tempo, party vibe, electronic dance music, tiktok trending",
    "house": "club-ready house remix inspired by {song_name}, four-on-the-floor kick, funky bassline, bright synth stabs, big drop, festival energy, clean mix",
    "techno": "driving techno reinterpretation inspired by {song_name}, 130 BPM, pounding kick, rolling bass, hypnotic synths, industrial atmosphere, peak-time energy",
    "slowed": "slowed and reverb version of {song_name}, vaporwave, chillout, atmospheric, lo-fi hip hop, sad vibes",
    "rnb": "2000s retro R&B reinterpretation inspired by {song_name}, smooth groove, rhodes piano, warm bass, silky drums, intimate vocal vibe, late-night city mood",
    "neosoul": "neo-soul / contemporary R&B reinterpretation inspired by {song_name}, laid-back pocket groove, rhodes chords, jazzy extensions, warm tape texture, emotional and intimate",
    "citypop": "retro city pop reinterpretation inspired by {song_name}, 80s Japanese city pop vibe, funky guitar, warm synths, tight drums, neon night drive mood, polished mix",
    "lofi": "lo-fi chill reinterpretation inspired by {song_name}, dusty drums, soft keys, vinyl crackle, mellow bass, nostalgic mood, cozy bedroom vibe",
    "lyrical": "lyrical emotional acoustic cover of {song_name}, beautiful piano melody, soft, sad, dramatic, expressive",
    "phonk": "dark aggressive phonk beat remix of {song_name}, hard distorted 808 bass, sharp cowbells, memphis rap atmosphere, high quality",
    "trap": "modern dark trap beat remix of {song_name}, punchy 808s, crisp hi hats with rolls, atmospheric pads, club ready mix",
    "synthwave": "retro 80s synthwave cover of {song_name}, analog synths, warm bass, big gated drums, neon city at night",
}

def get_config_dict():
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 初始化生成器单例，避免重复加载模型（如果可能）
generator = None

@app.on_event("startup")
def startup_event():
    global generator
    cfg_path = str(Path(__file__).resolve().parent / "config.yaml")
    generator = DailyPhonkGenerator(config_path=cfg_path)


def _build_prompt(song_name: str, style_key: str, extra_requirements: str | None) -> str:
    template = STYLE_TEMPLATES.get(style_key, STYLE_TEMPLATES["dj"])
    prompt = template.format(song_name=song_name)
    extra = (extra_requirements or "").strip()
    if extra:
        prompt = f"{prompt}. {extra}"
    return prompt


def _outputs_root_dir() -> str:
    raw_config = get_config_dict()
    out_cfg = raw_config.get("output", {})
    root = out_cfg.get("root_dir")
    if not root:
        raise HTTPException(status_code=500, detail="Output root_dir missing in config")
    return str(root)


@app.post("/generate")
async def generate_song(req: GenerateRequest, background_tasks: BackgroundTasks):
    if not req.song_name.strip():
        raise HTTPException(status_code=400, detail="Song name cannot be empty")

    style_key = req.style.lower()
    prompt = _build_prompt(req.song_name.strip(), style_key, req.extra_requirements)
    
    try:
        result = generator.generate_custom(
            prompt=prompt,
            style_id=f"remix_{style_key}",
            duration=req.duration,
            backend=req.backend
        )
        
        # 异步处理后续统计和生成附加资产
        raw_config = get_config_dict()
        background_tasks.add_task(generate_publishing_assets, raw_config, result)
        background_tasks.add_task(log_generation, raw_config, result)
        
        # 返回可以用来下载的文件名和信息
        return {
            "status": "success",
            "info": result,
            "download_url": f"/download?path={result['path']}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preview")
async def preview_song(req: PreviewRequest):
    if not req.song_name.strip():
        raise HTTPException(status_code=400, detail="Song name cannot be empty")

    style_key = req.style.lower()
    prompt = _build_prompt(req.song_name.strip(), style_key, req.extra_requirements)
    duration = int(req.duration or 15)
    candidates = int(req.candidates or 3)
    candidates = max(1, min(8, candidates))

    results = []
    for _ in range(candidates):
        seed = secrets.randbelow(2**31 - 1)
        try:
            result = generator.generate_custom(
                prompt=prompt,
                style_id=f"preview_{style_key}",
                duration=duration,
                backend=req.backend,
                seed=seed,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        results.append(
            {
                "seed": seed,
                "prompt": prompt,
                "info": result,
                "download_url": f"/download?path={result['path']}",
            }
        )

    return {
        "status": "success",
        "candidates": results,
    }


@app.post("/publish")
async def publish_song(req: PublishRequest, background_tasks: BackgroundTasks):
    if not req.song_name.strip():
        raise HTTPException(status_code=400, detail="Song name cannot be empty")

    style_key = req.style.lower()
    prompt = _build_prompt(req.song_name.strip(), style_key, req.extra_requirements)
    duration = int(req.duration or 45)
    seed = req.seed if req.seed is not None else secrets.randbelow(2**31 - 1)

    try:
        result = generator.generate_custom(
            prompt=prompt,
            style_id=f"remix_{style_key}",
            duration=duration,
            backend=req.backend,
            seed=seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    raw_config = get_config_dict()
    background_tasks.add_task(generate_publishing_assets, raw_config, result)
    background_tasks.add_task(log_generation, raw_config, result)

    base_name = os.path.splitext(result["path"])[0]
    meta_path = f"{base_name}_meta.json"
    promo_path = f"{base_name}_promo.txt"

    return {
        "status": "success",
        "seed": seed,
        "prompt": prompt,
        "info": result,
        "download_url": f"/download?path={result['path']}",
        "meta_url": f"/download?path={meta_path}",
        "promo_url": f"/download?path={promo_path}",
    }


@app.get("/download")
async def download_file(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    root_dir = os.path.realpath(_outputs_root_dir())
    real_path = os.path.realpath(path)
    if not (real_path == root_dir or real_path.startswith(root_dir + os.sep)):
        raise HTTPException(status_code=403, detail="Access denied")

    media_type, _ = mimetypes.guess_type(real_path)
    return FileResponse(real_path, media_type=media_type or "application/octet-stream", filename=os.path.basename(real_path))

@app.get("/styles")
async def get_styles():
    return {
        "styles": [
            {"id": "dj", "name": "DJ 舞曲版 (高燃)"},
            {"id": "slowed", "name": "慢速回响版 (氛围)"},
            {"id": "lyrical", "name": "抒情钢琴版 (走心)"},
            {"id": "phonk", "name": "暗黑 Phonk (经典)"},
            {"id": "trap", "name": "Trap 节奏 (说唱)"},
            {"id": "synthwave", "name": "合成器波 (复古80s)"},
        ]
    }
