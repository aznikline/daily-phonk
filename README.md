## Daily Phonk Music Generator

This project generates **one high‑quality electronic music track per day**, focusing on a **phonk** style, with occasional tracks in other related styles.

### Features

- **Phonk‑first & Beyond**: Generate phonk, trap, drum & bass, synthwave, DJ remixes, slowed versions, and lyrical covers.
- **Custom Song Inspiration (Web API)**: Input a song name and let the AI hallucinate a remix or cover in your chosen style.
- **Web UI & API**: Includes a modern TailwindCSS frontend and a FastAPI backend.
- **Neural backend (recommended)**: Uses Meta MusicGen via HuggingFace `transformers` to generate high‑quality, full electronic tracks from rich text prompts.
- **DSP fallback**: A lightweight built‑in synthesizer (pure `numpy` DSP) can be used if you disable the neural backend.
- **High audio quality**: Writes 32‑bit floating‑point WAV files at 44.1 kHz or higher.
- **Date‑based filenames**: Outputs are written under `outputs/YYYY/MM/DD/` with style in the filename.
- **Automated Publishing Assets**: Generates platform-standard metadata (JSON) and social media promo text (TXT) alongside the audio.
- **Analytics Tracking**: Automatically logs generation stats to a local SQLite database for analytics.

### Setup

1. **Create & activate a virtual environment (recommended):**

```bash
cd /Users/wizout/op/daily-phonk
python3 -m venv .venv
source .venv/bin/activate  # on macOS / Linux
```

2. **Install dependencies (with neural backend and Web API):**

```bash
pip install --upgrade pip
# 建议先单独安装 CPU 版 torch，避免某些平台解析问题
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> Note: The first run with the neural backend will download the MusicGen model weights (~GB level).

### Usage (Web UI & API)

To use the interactive web interface where you can input a song name and select a remix style:

1. **Start the API Backend**:
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
This will start a FastAPI server at `http://localhost:8000`.

2. **Open the Frontend**:
Open the file `frontend/index.html` in your browser, or serve it using any simple web server.

---

### Usage (CLI)

To generate a single track automatically (phonk most of the time, other styles occasionally):

```bash
cd /Users/wizout/op/daily-phonk
python -m src.main
```

You can also use command-line arguments to override configurations:

```bash
# Force the DSP backend for quick testing without model downloads
python -m src.main --backend dsp

# Set a specific random seed for reproducible generation
python -m src.main --seed 42

# Use a custom config file
python -m src.main --config /path/to/custom_config.yaml
```

By default this will:

- Choose a style according to configured probabilities.
- Generate ~\(duration\_seconds\) seconds of music (configurable).
- Save the track under `outputs/YYYY/MM/DD/STYLE_daily_phonk.wav`.
- Generate `..._meta.json` containing metadata for platforms (tags, artist, description, cover art prompt).
- Generate `..._promo.txt` containing formatted social media posts.
- Log the generation details to `outputs/stats.db`.

### Configuration

You can edit `src/config.yaml` to adjust:

- **backend**: `"neural"` (MusicGen via transformers) or `"dsp"` (built‑in synthesizer).
- **duration_seconds**: Target track length (used primarily by DSP backend; neural backend is controlled via `max_new_tokens`).
- **sample_rate**: Output sample rate.
- **styles** and **probabilities**: How often to generate phonk vs other styles.
- **prompts**: Text prompts for each style, which strongly affect the neural backend’s output.
- **publishing**: Standard format parameters (tags, descriptions, album name) for distribution and promo templates.
- **stats**: SQLite database path configuration for logging runs.
- **neural.model_name / device / guidance_scale / max_new_tokens**: Low‑level neural generation parameters.

### Vercel Deployment & Architecture
The project is designed with a **Separated Frontend & Backend** architecture:
1. **Frontend (`frontend/`)**: A pure HTML/CSS/JS frontend that can be hosted instantly on **Vercel** or **GitHub Pages**. Simply drag and drop the `frontend` folder into Vercel or link your GitHub repo.
   - *Important*: Once deployed, remember to update `const API_BASE` in `frontend/index.html` to point to your live backend IP/Domain instead of `localhost`.
2. **Backend (`src/api.py`)**: The FastAPI server that runs the actual PyTorch `MusicGen` AI model. This must be hosted on a machine with a GPU (e.g. RunPod, AWS EC2, or your local machine with `ngrok` port forwarding) because Vercel Serverless Functions do not support PyTorch or GB-sized model loading within a 10s timeout limit.

**To deploy the frontend to Vercel via CLI:**
```bash
cd /Users/wizout/op/daily-phonk/frontend
npm i -g vercel
vercel
```

---

### Automating Daily Generation (cron)

On macOS, you can add a cron job to run the generator once per day. For example:

```bash
crontab -e
```

Add a line like (runs at 03:00 every day):

```bash
0 3 * * * /bin/zsh -lc 'cd /Users/wizout/op/daily-phonk && source .venv/bin/activate && python -m src.main >> logs.txt 2>&1'
```

Make sure your virtual environment and Python path match your local setup.


