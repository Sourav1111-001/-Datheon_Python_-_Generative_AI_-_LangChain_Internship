from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
DATABASE_PATH = DATA_DIR / "documind.sqlite3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_FILE_SIZE_MB = 25
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 4
THEME_OPTIONS = ["Light", "Dark"]
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
