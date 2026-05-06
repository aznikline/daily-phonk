import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger("daily-phonk.metadata")

def generate_publishing_assets(config_dict: Dict[str, Any], generation_info: Dict[str, Any]) -> None:
    """
    生成各大平台发布所需的元数据、封面提示词及推广文案，并进行本地埋点统计。
    """
    pub_cfg = config_dict.get("publishing", {})
    if not pub_cfg:
        logger.warning("未找到 publishing 配置，跳过生成发布资产")
        return

    audio_path = generation_info["path"]
    base_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    style_id = generation_info["style_id"]
    date_str = generation_info["date"]
    prompt = generation_info["style_prompt"]
    
    # 1. 生成标准的元数据 JSON（供后续分发脚本读取）
    metadata = {
        "title": f"Daily Track - {date_str} ({style_id.title()})",
        "artist": pub_cfg.get("artist_name", "Daily Phonk Generator"),
        "album": pub_cfg.get("album_name", "Daily Drops"),
        "genre": style_id.title(),
        "tags": ["electronic", "daily", style_id, "ai-music"],
        "duration_seconds": generation_info["duration"],
        "backend": generation_info["backend"],
        "audio_file": os.path.basename(audio_path),
        "description": f"A daily generated {style_id} track.\nPrompt: {prompt}",
        "cover_art_prompt": pub_cfg.get("cover_prompt_prefix", "") + prompt.strip(),
    }
    
    # 2. 生成各平台推广文案
    social_templates = pub_cfg.get("social_templates", {})
    promo_texts = {}
    for platform, template in social_templates.items():
        promo_texts[platform] = template.format(
            style=style_id,
            date=date_str,
            prompt=prompt.strip()
        )
    metadata["social_promo"] = promo_texts

    # 3. 写入资产文件
    meta_path = os.path.join(base_dir, f"{base_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    # 生成独立的文案文件，方便直接复制
    promo_path = os.path.join(base_dir, f"{base_name}_promo.txt")
    with open(promo_path, "w", encoding="utf-8") as f:
        f.write("=== COVER ART PROMPT ===\n")
        f.write(metadata["cover_art_prompt"] + "\n\n")
        for platform, text in promo_texts.items():
            f.write(f"=== {platform.upper()} ===\n")
            f.write(text + "\n\n")
            
    logger.info(f"已生成发布元数据及推广文案: {meta_path}")

