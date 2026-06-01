import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # AWS stuff - needed for Bedrock API calls
    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str

    # which model to use for generating answers
    bedrock_model_id: str

    # local embedding model, runs on your machine
    embedding_model_name: str

    # settings for splitting PDF text into chunks
    chunk_size: int
    chunk_overlap: int

    # how many chunks to send to the LLM per question
    top_k_results: int

    # server settings
    api_host: str
    api_port: int


def load_config() -> Config:
    """
    Read and validate all environment variables at startup.

    I centralized this here so the app fails with a clear error
    if something is missing - instead of crashing randomly later.
    """
    missing = []

    for var in ["AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return Config(
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k_results=int(os.getenv("TOP_K_RESULTS", "5")),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
    )


# single global instance - everything else imports from here
config = load_config()
