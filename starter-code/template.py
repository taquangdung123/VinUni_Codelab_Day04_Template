"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO (DONE) 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Vai trò: Tư vấn viên sản phẩm VinFast, dịch vụ Vinpearl và hỗ trợ khách hàng.
- Phong cách: Chuyên nghiệp, thân thiện, ngắn gọn và chính xác.

## AVAILABLE TOOLS
- search_product_catalog: Tra cứu sản phẩm hoặc dịch vụ theo danh mục và giá tối đa.
- submit_support_ticket: Tạo yêu cầu hỗ trợ cho khách hàng.

## CORE RULES
1. Không bịa thông tin sản phẩm, giá, tình trạng hoặc mã ticket.
2. Luôn gọi tool tương ứng khi câu hỏi cần dữ liệu catalog hoặc cần tạo ticket.
3. Chỉ dùng dữ liệu trả về từ tool để đưa ra kết luận.
4. Nếu không có kết quả, phải nói rõ rằng không tìm thấy dữ liệu phù hợp.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ sản phẩm, dịch vụ và yêu cầu thuộc hệ sinh thái Vingroup.
- Với câu hỏi ngoài phạm vi, lịch sự thông báo giới hạn hỗ trợ.

## OUTPUT CONTRACT
Trả lời theo cấu trúc: Thought -> Action -> Observation -> Final Answer.
Final Answer phải bằng tiếng Việt và nêu rõ kết quả hoặc mã ticket nếu có.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO (DONE) 2: Trả về mock response, không sử dụng tool calling.
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        # TODO (DONE) 3: Phân tích độc lập intent catalog và ticket.
        lowered_input = user_input.lower()
        catalog_keywords = ("xe điện", "vinfast", "resort", "vinpearl", "du lịch")
        catalog_request_keywords = (
            "xem", "tìm", "giá", "dưới", "danh sách", "sản phẩm nào", "phòng nào"
        )
        ticket_keywords = ("hỗ trợ", "phản hồi", "khiếu nại", "bị lỗi", "lỗi", "ticket")
        has_customer_name = re.search(
            r"tên\s+(?:tôi\s+)?(?:là\s+)?[^,.;]+",
            user_input,
            flags=re.IGNORECASE,
        )
        needs_catalog = (
            any(keyword in lowered_input for keyword in catalog_keywords)
            and any(keyword in lowered_input for keyword in catalog_request_keywords)
        )
        needs_ticket = any(keyword in lowered_input for keyword in ticket_keywords) and bool(has_customer_name)
        intents = {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": not needs_catalog and not needs_ticket,
        }
        self.trace.append({"step": "intent_detection", "intents": intents})

        # TODO (DONE) 4: Thực thi lần lượt các tool cần thiết và tổng hợp câu trả lời.
        if self.max_iterations < 1:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached",
            }

        tool_requests = []
        if needs_catalog:
            tool_requests.append(("search_product_catalog", self._catalog_arguments(user_input)))
        if needs_ticket:
            tool_requests.append(("submit_support_ticket", self._ticket_arguments(user_input)))

        observations = []
        for iteration, (tool_name, arguments) in enumerate(tool_requests, start=1):
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration - 1,
                    "status": "max_iterations_reached",
                }

            self.trace.append({
                "step": "tool_call",
                "iteration": iteration,
                "tool": tool_name,
                "arguments": arguments,
            })
            result = TOOL_MAP[tool_name](**arguments)
            observations.append((tool_name, result))
            self.trace.append({
                "step": "observation",
                "iteration": iteration,
                "tool": tool_name,
                "result": result,
            })

        answer = self._build_answer(user_input, intents, observations)
        self.trace.append({"step": "final_answer", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": len(observations) if observations else 1,
            "status": "completed",
        }

    @staticmethod
    def _catalog_arguments(user_input: str) -> Dict[str, Any]:
        lowered_input = user_input.lower()
        category = "du_lich" if any(
            keyword in lowered_input for keyword in ("du lịch", "resort", "vinpearl")
        ) else "xe_dien"
        max_price = 999999999999
        price_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|tỉ|million|m)",
            lowered_input,
            flags=re.IGNORECASE,
        )
        if price_match:
            amount = float(price_match.group(1).replace(",", "."))
            unit = price_match.group(2).lower()
            multiplier = 1000000000 if unit in ("tỷ", "tỉ") else 1000000
            max_price = int(amount * multiplier)
        return {"category": category, "max_price": max_price}

    @staticmethod
    def _ticket_arguments(user_input: str) -> Dict[str, Any]:
        name_match = re.search(
            r"tên\s+(?:tôi\s+)?(?:là\s+)?([^,.;]+)",
            user_input,
            flags=re.IGNORECASE,
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"
        issue_description = user_input[name_match.end():].strip(" ,.;:") if name_match else user_input.strip()
        issue_description = re.sub(
            r"^(?:và\s+)?(?:tôi\s+)?(?:cũng\s+)?muốn\s+(?:ghi\s+nhận\s+)?(?:phản\s+hồi\s*:)?\s*",
            "",
            issue_description,
            flags=re.IGNORECASE,
        )
        issue_description = re.split(
            r",\s*(?:mức độ|mức ưu tiên)|\.\s*(?:đây là|cần xử lý)|,\s*cần xử lý",
            issue_description,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,.;")
        if issue_description:
            issue_description = issue_description[0].upper() + issue_description[1:]

        lowered_input = user_input.lower()
        priority = "high" if any(
            keyword in lowered_input for keyword in ("nghiêm trọng", "gấp", "khẩn", "urgent")
        ) else "low" if "thấp" in lowered_input else "medium"
        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority,
        }

    @staticmethod
    def _build_answer(
        user_input: str,
        intents: Dict[str, bool],
        observations: List[Any],
    ) -> str:
        if intents["is_faq"]:
            if "bảo hành" in user_input.lower() and "pin" in user_input.lower():
                return "Pin xe điện VinFast được bảo hành 10 năm theo dữ liệu sản phẩm hiện có."
            return "VinAssistant chỉ hỗ trợ thông tin sản phẩm, dịch vụ và yêu cầu hỗ trợ thuộc Vingroup."

        answer_parts = []
        for tool_name, result in observations:
            if tool_name == "search_product_catalog":
                if not result:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    names = ", ".join(product["name"] for product in result if "name" in product)
                    answer_parts.append(f"Sản phẩm phù hợp: {names}.")
            elif tool_name == "submit_support_ticket":
                answer_parts.append(
                    f"Đã tạo ticket {result['ticket_id']} cho khách hàng {result['customer_name']}."
                )
        return " ".join(answer_parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
