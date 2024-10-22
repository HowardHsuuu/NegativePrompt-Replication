import time
import requests
import os

from dotenv import load_dotenv
from utility import progress_bar

from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import torch

load_dotenv()

llm_openai = ['gpt-3.5-turbo', 'gpt-4', 'vicuna']
llm_hf = ['llama2', 't5', 'phi-3.5-mini', 'qwen2.5']

MODEL_MAPPING = {
    'phi-3.5-mini': 'microsoft/Phi-3.5-mini-instruct',
    't5': 'google/flan-t5-large',
    'llama2': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3.2': 'meta-llama/Llama-3.2-1B',
    'qwen2.5': 'Qwen/Qwen2.5-0.5B-Instruct',
    'gpt-3.5': 'gpt-3.5-turbo',
    'gpt-4o': 'gpt-4o',
    'vicuna': 'vicuna-13b-v1.1'  # Assuming this is the correct model name for Vicuna
}

def get_response_from_llm(llm_id: str, instruction: str, queries: list[str], api_num=4, local=True):
    if llm_id in llm_openai:
        print("Connecting with OpenAI API...")
        return get_openai_response(llm_id, queries)
    elif llm_id in llm_hf:
        if local:
            print("Accessing local instance of model...")
            return get_local_hf_response(llm_id, instruction, queries)
        else:
            print("Connecting with Hugging Face API...")
            return get_api_hf_response(llm_id, queries, api_num)
    else:
        raise ValueError('llm_id is not supported')

def get_openai_response(llm_id: str, queries: list[str]):
    client = OpenAI()
    model_outputs = []

    for query in queries:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=llm_id,
                    messages=[{"role": "user", "content": query}],
                    temperature=0.7,
                    max_tokens=200
                )
                output = response.choices[0].message.content.strip()
                model_outputs.append(output)
                print(f"Query: {query}\nOutput: {output}\n")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    model_outputs.append("")
                else:
                    print(f"Attempt {attempt + 1} failed. Retrying... Error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff

    return model_outputs

def get_local_hf_response(llm_id: str, instruction: str, queries: list[str]):
    model_outputs = []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[llm_id], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_MAPPING[llm_id], torch_dtype="auto", device_map="auto", trust_remote_code=True)
    device = model.device

    for i, query in enumerate(queries):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=200)

        # Process the output by skipping the input tokens and special tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        model_outputs.append(output)
        # print(f"Query: {query}\nOutput: {output}\n")
        progress_bar(i, len(queries)-1, prefix='Generating Model Predictions', length=50)
        

    return model_outputs

def get_api_hf_response(llm_id: str, queries: list[str], api_num=4):
    API_URL = f"https://api-inference.huggingface.co/models/{llm_id}"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
    model_outputs = []

    def query_api(payload):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"API request failed after {max_retries} attempts. Error: {e}")
                    return None
                else:
                    print(f"Attempt {attempt + 1} failed. Retrying... Error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff

    for query in queries:
        payload = {"inputs": query}
        result = query_api(payload)
        
        if result:
            output = result[0]['generated_text'].strip()
            model_outputs.append(output)
            print(f"Query: {query}\nOutput: {output}\n")
        else:
            model_outputs.append("")

    return model_outputs