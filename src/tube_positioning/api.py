from pathlib import Path
from fastapi import FastAPI
from tube_positioning import __version__
from tube_positioning.config import load_config

app = FastAPI(title="Tube Positioning API", version=__version__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "motion_authority": False}


@app.get("/config")
def config():
    return load_config(DEFAULT_CONFIG)
