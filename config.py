
from pathlib import Path

COURT_ID = "cand"
CASE_TYPE_KEYWORDS = ["Contract"]
CL_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
CL_DOCKETS_ENDPOINT = f"{CL_BASE_URL}/dockets/"
CL_CLUSTERS_ENDPOINT = f"{CL_BASE_URL}/clusters/"
CL_OPINIONS_ENDPOINT = f"{CL_BASE_URL}/opinions/"
CL_REQUEST_TIMEOUT_SECONDS = 60

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

RAW_DOCKETS_PATH = RAW_DIR / "dockets.jsonl"
RAW_CLUSTERS_PATH = RAW_DIR / "clusters.jsonl"
RAW_OPINIONS_PATH = RAW_DIR / "opinions.jsonl"
