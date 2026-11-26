"""vLLM client, prompt loading and JSON parsing for the prompting experiments."""

import json
import os

from openai import OpenAI

# Each model is served by vLLM behind an OpenAI-compatible endpoint.
# Set VLLM_BASE_URL_<MODEL> per model, or VLLM_BASE_URL for all of them.
DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
GPU_URLS = {
    name: os.environ.get(f"VLLM_BASE_URL_{name.upper()}", DEFAULT_BASE_URL)
    for name in ("qwen", "gemma", "mistral", "gpt")
}
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

model_maps = {
    "gpt": "openai/gpt-oss-20b",
    "mistral": "mistralai/Magistral-Small-2509", 
    "gemma": "google/gemma-4-31B-it", 
    "qwen": "Qwen/Qwen3.5-35B-A3B"
}

prompt_maps = {
    "binary": {
        "A": "prompts/binary/A.json",
        "B": "prompts/binary/B.json",
        "C": "prompts/binary/C.json",
        "D": "prompts/binary/D.json",
    },
    "multiclass": {
        "A": "prompts/multiclass/A.json",
        "B": "prompts/multiclass/B.json",
        "C": "prompts/multiclass/C.json",
        "D": "prompts/multiclass/D.json",
    }
}

client = None
model_name = None
user_prompt = None
user_shots_prompt = None
system_prompt = None
system_shots_prompt = None

def set_model_and_prompt(model: str, task: str, prompt: str):
    global model_name, client, user_shots_prompt, user_prompt, system_prompt, system_shots_prompt
    model_name = model_maps[model]
    client = OpenAI(
        base_url=GPU_URLS[model],
        api_key=VLLM_API_KEY
    )
    with open(prompt_maps[task][prompt], "r", encoding="utf-8") as f:
        data = json.load(f)
    user_shots_prompt = data["user_shots_prompt"]
    user_prompt = data["user_prompt"]
    if model == "mistral":
        system_prompt = """First draft your thinking process (inner monologue) until you arrive at a response. Write both your thoughts and the response in English language.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.
\n\n""" + data["system_prompt"]
        system_shots_prompt = """First draft your thinking process (inner monologue) until you arrive at a response. Write both your thoughts and the response in English language.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.
\n\n""" + data["system_shots_prompt"]
    elif model == "gemma":
        system_prompt = "<|think|>You are a reasoning assistant. First draft your thinking process. Then respond.\n\n" + data["system_prompt"]
        system_shots_prompt = "<|think|>You are a reasoning assistant. First draft your thinking process. Then respond.\n\n" + data["system_shots_prompt"]
    else:
        system_prompt = data["system_prompt"]
        system_shots_prompt = data["system_shots_prompt"]

def call_vllm_model(model: str = model_name, user_prompt: str = "", system_prompt: str = "", max_retries: int = 5):
    """Call the vLLM OpenAI-compatible endpoint with retry logic."""

    global client

    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            max_tokens = 25000

            model_response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,     # max output tokens (len(prompt_tokens) + max_tokens ≤ max_model_len)
                temperature=0,             # greedy decoding
            )
            
            lines = model_response.choices[0].message.content.strip().splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)
            
            cot = model_response.choices[0].message.reasoning

            # Extract if parser failed
            if not cot:
                # The server did not split reasoning from content: do it by hand
                close_idx = response.rfind('}')

                if close_idx == -1:
                    # No '}' found
                    pass
                else:

                    # Find the last '{' before that '}'
                    open_idx = response.rfind('{', 0, close_idx)

                    if open_idx == -1:
                        # No '{' before the '}'
                        pass
                    
                    else:

                        # JSON block is the answer, the text before it is the reasoning
                        _response = response[open_idx:close_idx + 1]
                        _cot = response[:open_idx]
                        response = _response
                        cot = _cot

            if not cot and not response:
                raise ValueError("Missing CoT and response content")
            
            # Success
            if attempt > 0:
                print(f" Succeeded on attempt {attempt + 1}")
            
            return response, cot, model_response

        except Exception as e:
            print(f" Generic Error - attempt {attempt + 1}/{max_retries}: {e}")
            
            if attempt == max_retries - 1:
                print(f" All {max_retries} attempts failed, returning None.")
                return None, "", response
            else:
                print(" Retrying...")
    return None, "", ""

def call_model_binary(
    shots: str = "",
    report: str = "",
    model: str = "",
):
    """Binary classification: returns {'Valutazione': ..., 'Motivazione': ...}"""

    global user_prompt, user_shots_prompt, system_prompt

    max_retries = 5

    for attempt in range(max_retries):
        try:
            if shots:
                system = system_shots_prompt
                user = user_shots_prompt.format(report=report, shots=shots)
            else:
                system = system_prompt
                user = user_prompt.format(report=report)

            response, cot, full_response = call_vllm_model(model=model_name, user_prompt=user, system_prompt=system)
            json_response = json.loads(response) if response else None
            if attempt > 0:
                print(f"Succeeded on attempt {attempt + 1}")
            return json_response, cot, full_response

        except Exception as e:
            print(f"Generic Error - attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed, returning None.")
                return None, "", full_response
            else:
                print("Retrying...")

    return None, "", ""


def call_model_multiclass(
    shots: str = "",
    report: str = "",
    model: str = "",
):
    """Multiclass classification: returns {'Categorie': ..., 'Motivazione': ...}"""

    global user_prompt, user_shots_prompt, system_prompt

    max_retries = 5

    for attempt in range(max_retries):
        try:
            if shots:
                system = system_shots_prompt
                user = user_shots_prompt.format(report=report, shots=shots)
            else:
                system = system_prompt
                user = user_prompt.format(report=report)

            response, cot, full_response = call_vllm_model(model=model_name, user_prompt=user, system_prompt=system)
            json_response = json.loads(response) if response else None
            if attempt > 0:
                print(f"Succeeded on attempt {attempt + 1}")
            return json_response, cot, full_response

        except Exception as e:
            print(f"Generic Error - attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed, returning None.")
                return None, "", full_response
            else:
                print("Retrying...")

    return None, "", ""