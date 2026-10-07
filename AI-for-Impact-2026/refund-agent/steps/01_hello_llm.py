"""Bước 1: gọi LLM một lần. Đây là "hello world" thật sự.

Chạy: uv run python steps/01_hello_llm.py
Cần biến môi trường ANTHROPIC_API_KEY.
"""

import anthropic
import truststore

import os

truststore.inject_into_ssl()  # tin CA của hệ điều hành (hữu ích sau proxy công ty)

MODEL = "google/gemini-3.8-flash"

base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
client = anthropic.Anthropic(base_url=base_url, api_key=api_key)

response = client.messages.create(
    model=MODEL,
    max_tokens=1500,
    messages=[{"role": "user", "content": "Chính sách hoàn tiền thường gồm những ý chính nào?"}],
)

text = "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text")
print(text)
print("---")
print("stop_reason:", response.stop_reason)
print("token vào/ra:", response.usage.input_tokens, "/", response.usage.output_tokens)
