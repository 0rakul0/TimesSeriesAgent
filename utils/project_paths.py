from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "eval"
IMG_DIR = BASE_DIR / "img"
MODELOS_DIR = BASE_DIR / "modelos"
OUTPUT_NOTICIAS_DIR = BASE_DIR / "output_noticias"
SRC_DIR = BASE_DIR / "src"
TRAIN_DIR = BASE_DIR / "train"
UTILS_DIR = BASE_DIR / "utils"


def repo_path(*parts: str) -> Path:
    return BASE_DIR.joinpath(*parts)


def ensure_runtime_dirs() -> None:
    for directory in (IMG_DIR, OUTPUT_NOTICIAS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
