"""Bước 4: vòng lặp agent tối giản, với 2 tool.

Biến bước 3 thành agent bằng một vòng `for`: lặp đến khi LLM không đòi gọi tool nữa,
hoặc chạm giới hạn số bước.

Chạy: uv run python steps/04_agent_loop.py
"""

import json
from typing import Any, cast

import anthropic
import truststore

from refund_agent.shop import Shop

import os

truststore.inject_into_ssl()

MODEL = "google/gemini-3.8-flash"
MAX_STEPS = 6
SYSTEM = "Bạn là trợ lý hoàn tiền. Dùng tool để kiểm tra trước khi kết luận. Trả lời tiếng Việt."
shop = Shop()

base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
client = anthropic.Anthropic(base_url=base_url, api_key=api_key)

ORDER_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}, "customer_email": {"type": "string"}},
    "required": ["order_id", "customer_email"],
}
TOOLS: list[anthropic.types.ToolParam] = [
    {"name": "get_order", "description": "Tra cứu đơn hàng.", "input_schema": ORDER_PARAMS},
    {
        "name": "check_refund_policy",
        "description": "Kiểm tra đơn có đủ điều kiện hoàn tiền không.",
        "input_schema": ORDER_PARAMS,
    },
]


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    order = shop.find_order(args["order_id"], args["customer_email"])
    if order is None:
        return {"error": "Không tìm thấy đơn hàng khớp mã đơn và email."}
    if name == "get_order":
        return {"item": order.item, "amount": order.amount, "status": order.status}
    if name == "check_refund_policy":
        return shop.evaluate_policy(order)
    return {"error": f"Không có tool {name}"}


messages: list[anthropic.types.MessageParam] = [
    {"role": "user", "content": "Tôi muốn hoàn tiền đơn DH1002, email binh@example.com."}
]

for step in range(1, MAX_STEPS + 1):
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        tools=TOOLS,
        messages=messages,
    )
    print(f"--- Bước {step}: stop_reason={response.stop_reason}")
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":  # LLM nói xong rồi
        answer = "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        print(answer)
        break

    results: list[anthropic.types.ToolResultBlockParam] = []
    for block in response.content:
        if block.type == "tool_use":
            output = run_tool(block.name, cast(dict[str, Any], block.input))
            print(f"    gọi {block.name}({block.input}) -> {output}")
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, ensure_ascii=False),
                }
            )
    messages.append({"role": "user", "content": results})
else:
    print("Chạm giới hạn số bước mà chưa xong.")
