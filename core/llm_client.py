"""LLM clients for production Bedrock calls and deterministic demo mode."""

import json
from typing import Protocol

import boto3


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate an answer for a fully assembled RAG prompt."""


class DemoLLMClient:
    """Deterministic local client that never calls an external service."""

    def generate(self, prompt: str) -> str:
        context_marker = "Document context:\n"
        question_marker = "\n\nQuestion:"
        context = prompt.split(context_marker, 1)[-1].split(question_marker, 1)[0]
        first_source = next(
            (line.strip() for line in context.splitlines() if line.strip()),
            "No document context was retrieved.",
        )
        return f"[Demo mode — no external LLM call] Relevant context: {first_source}"


class BedrockLLMClient:
    """Amazon Bedrock Claude client used when demo mode is disabled."""

    def __init__(self, config):
        self.model_id = config.bedrock_model_id
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )

    def generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]


def build_llm_client(config) -> LLMClient:
    """Choose a no-credential demo client or the production Bedrock client."""
    if config.demo_mode:
        return DemoLLMClient()
    return BedrockLLMClient(config)
