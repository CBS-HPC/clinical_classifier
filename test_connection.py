#test_connection.py
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=False)

OPENAI_API_BASE = "https://app-clinical-trial.cloud.aau.dk/v1"
OPENAI_API_KEY  = "token-99898157f592e2bd278a49fa1a4a0ee3"


if not OPENAI_API_BASE or not OPENAI_API_KEY:
    print(
        "Missing OPENAI_API_BASE and/or OPENAI_API_KEY.\n"
        "Create a .env file (or set real env vars) with:\n\n"
        "OPENAI_API_BASE\n"
        "OPENAI_API_KEY\n",
        file=sys.stderr,
    )
    sys.exit(1)

client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

# List models and export the first one (like your original script)
try:
    models = client.models.list()
    OPENAI_MODEL = models.data[0].id
except Exception as e:
    print(f"Failed to list models: {e}", file=sys.stderr)
    sys.exit(2)

os.environ["OPENAI_MODEL"] = OPENAI_MODEL
print(OPENAI_MODEL)
