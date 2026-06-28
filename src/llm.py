import boto3
import json

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "amazon.nova-lite-v1:0"


def generate_answer(question, context):

    prompt = f"""
You are an Enterprise Financial Intelligence Assistant.

Answer ONLY using the information provided below.

If the answer is not present in the context, reply:
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
        ],
        "inferenceConfig": {
            "temperature": 0.2,
            "max_new_tokens": 512
        }
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    response_body = json.loads(
        response["body"].read()
    )

    return response_body["output"]["message"]["content"][0]["text"]