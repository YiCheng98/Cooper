import requests
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential

with open("my_api_key.txt", "r") as f:
    my_api_key = f.read().strip()

# Define the function to call the LLM API. It returns the generated text from LLM.
@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
def call_LLM(prompt, api_key=my_api_key, max_output_tokens = 512, model="gpt-3.5-turbo"):
    url = "https://api.openai-sb.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_output_tokens,
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    response_json = json.loads(response.text)
    content = response_json["choices"][0]["message"]["content"]
    return content

@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
def call_embedding_model(text, api_key=my_api_key, model="text-embedding-ada-002"):
    url = "https://api.openai-sb.com/v1/embeddings"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "input": text
    }

    response = requests.post(url, headers=headers, json=data)
    response_json = json.loads(response.text)
    content = response_json["data"][0]["embedding"]
    return content