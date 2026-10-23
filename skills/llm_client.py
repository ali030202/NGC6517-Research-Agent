"""Optional OpenAI-compatible client; credentials are never loaded at import."""
import json
import logging
import os
import re
import time

MAX_NUM_TOKENS = 8192


class LLMNotConfigured(ValueError):
    pass


def create_llm_client(model=None, base_url=None, api_key=None):
    model = model or os.getenv("AGENT_MODEL") or os.getenv("DASHSCOPE_MODEL")
    base_url = base_url or os.getenv("AGENT_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
    api_key = api_key or os.getenv("AGENT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not all((model, base_url, api_key)):
        raise LLMNotConfigured("API 功能待接入：需要显式提供模型、地址和凭据。本地核心无需这些配置。")
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0), model


def get_response_from_llm(msg, client, model, system_message, print_debug=False,
                          msg_history=None, temperature=0.75, enable_thinking=False):
    history = list(msg_history or []) + [{"role": "user", "content": msg}]
    kwargs = {"extra_body": {"enable_thinking": True}} if enable_thinking else {}
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": system_message}, *history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, stream=False, **kwargs,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LLM returned no text")
            if getattr(response.choices[0], "finish_reason", None) == "length":
                raise ValueError("LLM output was truncated; no partial code will be applied")
            return content, history + [{"role": "assistant", "content": content}]
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status in (408, 409, 429) or (isinstance(status, int) and status >= 500) or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}
            if not retryable or attempt == 2:
                # Do not include URLs, headers, keys, prompts or provider response bodies.
                raise RuntimeError(f"LLM request failed ({type(exc).__name__}, status={status})") from None
            logging.getLogger(__name__).warning("LLM request temporarily unavailable; retry %s/2", attempt + 1)
            time.sleep(2 ** (attempt + 1))


def get_batch_responses_from_llm(msg, client, model, system_message, print_debug=False,
                                msg_history=None, temperature=0.75, n_responses=1):
    if not 1 <= n_responses <= 8:
        raise ValueError("n_responses must be between 1 and 8")
    responses = [get_response_from_llm(msg, client, model, system_message,
                 msg_history=msg_history, temperature=temperature) for _ in range(n_responses)]
    return [item[0] for item in responses], [item[1] for item in responses]


def extract_json_between_markers(llm_output):
    decoder = json.JSONDecoder()
    for content in [*re.findall(r"```json\s*(.*?)```", llm_output, re.S), llm_output]:
        content = content.strip()
        for index, character in enumerate(content):
            if character not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(content[index:])
                if isinstance(value, (dict, list)):
                    return value
            except json.JSONDecodeError:
                continue
    return None


def quick_llm_call(prompt, system_message="You are a helpful AI research assistant.", model=None, enable_thinking=False):
    client, selected_model = create_llm_client(model=model)
    return get_response_from_llm(prompt, client, selected_model, system_message,
                                 enable_thinking=enable_thinking)[0]
