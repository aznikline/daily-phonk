import sqlite3
import os
import logging
from typing import Dict, Any

logger = logging.getLogger("daily-phonk.stats")

def init_db(db_path: str) -> None:
    """初始化统计数据库表"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            style_id TEXT NOT NULL,
            backend TEXT NOT NULL,
            duration INTEGER,
            file_path TEXT,
            published INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_generation(config_dict: Dict[str, Any], generation_info: Dict[str, Any]) -> None:
    """
    记录单次生成的统计信息（埋点）
    """
    stats_cfg = config_dict.get("stats", {})
    if not stats_cfg:
        logger.warning("未找到 stats 配置，跳过统计记录")
        return

    db_path = stats_cfg.get("db_path", "outputs/stats.db")
    init_db(db_path)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO generations (date, style_id, backend, duration, file_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            generation_info["date"],
            generation_info["style_id"],
            generation_info["backend"],
            generation_info["duration"],
            generation_info["path"]
        ))
        conn.commit()
        conn.close()
        logger.info(f"生成数据已写入统计数据库: {db_path}")
    except Exception as e:
        logger.error(f"写入统计数据库失败: {e}")
