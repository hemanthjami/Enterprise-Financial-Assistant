import boto3
import json

# Create Bedrock Runtime client
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "amazon.nova-lite-v1:0"


def generate_answer(question, context):

    prompt = f"""
You are a Financial Business Intelligence Assistant.

Use ONLY the information provided below to answer the user's question.

If the answer is not found in the context, reply:
"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())

    return result["output"]["message"]["content"][0]["text"]