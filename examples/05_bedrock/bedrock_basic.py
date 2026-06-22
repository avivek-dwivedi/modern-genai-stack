"""
examples/05_bedrock/bedrock_basic.py

Basic AWS Bedrock invocation with Claude.
Maps to docs/10-aws-production/README.md "Bedrock vs Self-Hosted".

NOTE: Install dependencies before running:
    pip install boto3
    aws configure
"""

import os
import json
import boto3


# ---------- Configuration ----------

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # swap for your model


def get_bedrock_client():
    """Build a Bedrock runtime client."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION,
    )


def invoke_claude(client, prompt: str, max_tokens: int = 1024) -> str:
    """Invoke Claude on Bedrock."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        "temperature": 0.2,
    })

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


def invoke_with_guardrail(client, prompt: str, guardrail_id: str, guardrail_version: str) -> str:
    """Invoke Claude with a Bedrock Guardrail applied."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    })

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
    )

    return json.loads(response["body"].read())


def invoke_with_rag(client, prompt: str, knowledge_base_id: str) -> str:
    """
    Invoke Claude with RetrieveAndGenerate (Bedrock Knowledge Base RAG).
    Maps to docs/07-ragops/README.md "Production RAGOps Pipeline".
    """
    rag_client = boto3.client("bedrock-agent-runtime", region_name=REGION)

    response = rag_client.retrieve_and_generate(
        input={"text": prompt},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": knowledge_base_id,
                "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/{MODEL_ID}",
            },
        },
    )

    return response["output"]["text"]


# ---------- Run ----------

if __name__ == "__main__":
    client = get_bedrock_client()

    prompt = "Explain LoRA in one sentence."
    print(f"Prompt: {prompt}")
    print(f"Answer: {invoke_claude(client, prompt)}")