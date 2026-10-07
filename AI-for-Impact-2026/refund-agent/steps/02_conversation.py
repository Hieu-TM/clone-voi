"""Bước 2: system prompt và hội thoại nhiều lượt.

Điểm cần thấy: LLM KHÔNG nhớ gì. Mỗi lần gọi ta gửi lại TOÀN BỘ lịch sử.

Chạy: uv run python steps/02_conversation.py
"""

import anthropic
import truststore

import os

truststore.inject_into_ssl()

MODEL = "google/gemini-3.8-flash"
SYSTEM = (
    "Bạn là trợ lý chăm sóc khách hàng của cửa hàng online. "
    "Trả lời tiếng Việt, lịch sự, ngắn gọn. Chưa có công cụ nào, nên đừng giả vờ tra cứu đơn."
)

base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
client = anthropic.Anthropic(base_url=base_url, api_key=api_key)
history: list[anthropic.types.MessageParam] = []


def say(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    response = client.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM, messages=history)
    answer = "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    history.append({"role": "assistant", "content": answer})
    print(f"Khách: {user_text}\nBot  : {answer}\n(lịch sử hiện có {len(history)} message)\n")
    return answer


say("Tôi muốn hoàn tiền cái tai nghe vừa mua.")
say("Mã đơn của tôi là DH1001.")  # nhờ có lịch sử, bot biết "đơn" này nói về tai nghe
