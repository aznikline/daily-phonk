from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import yaml
from typing import Optional
from pathlib import Path

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

# 定义不同风格的模板提示词
STYLE_TEMPLATES = {
    "dj": "high energy DJ club remix of {song_name}, heavy EDM bass, fast tempo, party vibe, electronic dance music, tiktok trending",
    "slowed": "slowed and reverb version of {song_name}, vaporwave, chillout, atmospheric, lo-fi hip hop, sad vibes",
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

@app.post("/generate")
async def generate_song(req: GenerateRequest, background_tasks: BackgroundTasks):
    if not req.song_name.strip():
        raise HTTPException(status_code=400, detail="Song name cannot be empty")

    style_key = req.style.lower()
    template = STYLE_TEMPLATES.get(style_key, STYLE_TEMPLATES["dj"])
    prompt = template.format(song_name=req.song_name)
    
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

@app.get("/download")
async def download_file(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(path))

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
