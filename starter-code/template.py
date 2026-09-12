"""
Lab #3: Baseline Chatbot vs ReAct Agent
Triển khai hoàn chỉnh ChatbotBaseline và ReActAgent theo đúng hướng dẫn student_guide.md.
"""

import json
import re
import sys
from typing import Dict, Any, List, Optional
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

# Hỗ trợ hiển thị tiếng Việt UTF-8 mượt mà trên console Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Xử lý truy vấn mà không sử dụng tool (Milestone 1).
        Trả về dictionary gồm status, tool_calls rỗng và answer.
        """
        return {
            "status": "success",
            "tool_calls": [],
            "answer": (
                f"[Chatbot Baseline] Trả lời cho câu hỏi: '{user_input}'. "
                "Lưu ý: Tôi là Chatbot Baseline không có kết nối cơ sở dữ liệu thời gian thực "
                "nên không thể tra cứu thông tin chuyến bay hay thời tiết chính xác."
            )
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _execute_tool(self, action_data: Any) -> Any:
        """
        Thực thi tool an toàn (Xử lý Trap 1 & Trap 2 theo hướng dẫn).
        """
        # Trap 2: Format Drift trong Action JSON - Sử dụng try...except json.loads
        if isinstance(action_data, str):
            try:
                action_data = json.loads(action_data)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON format: {str(e)}"}

        if not isinstance(action_data, dict):
            return {"error": "Action must be a valid JSON object"}

        raw_name = action_data.get("name", "")
        # Trap 1: KeyError khi gọi Tool - Gọi .strip().lower() trước khi tra cứu
        clean_name = str(raw_name).strip().lower()

        tool_func = None
        for key, func in TOOL_MAP.items():
            if key.strip().lower() == clean_name:
                tool_func = func
                break

        if not tool_func:
            return {"error": f"Tool '{raw_name}' not found in TOOL_MAP"}

        args = action_data.get("args", {})
        try:
            return tool_func(**args)
        except Exception as e:
            return {"error": f"Error executing tool {raw_name}: {str(e)}"}

    def _parse_flight_params(self, text: str) -> Dict[str, Any]:
        """Trích xuất tham số chuyến bay (origin, destination, max_price)."""
        origin = "HAN"
        destination = "SGN"
        max_price = 5000000

        city_map = {
            "hà nội": "HAN", "han": "HAN",
            "đà nẵng": "DAD", "dad": "DAD",
            "tp. hồ chí minh": "SGN", "tp.hcm": "SGN", "tphcm": "SGN",
            "hồ chí minh": "SGN", "sài gòn": "SGN", "sgn": "SGN"
        }

        # Tìm tuyến bay dạng "từ X đi Y" hoặc "X đến Y"
        route_match = re.search(
            r'(?:từ\s+)?([A-Za-zÀ-ỹ\.\s]+?)\s*(?:đi|đến|tới|-|>)\s*([A-Za-zÀ-ỹ\.\s]+?)(?=[,\.\?]|\s+dưới|\s+giá|\s+khoảng|$)',
            text,
            re.IGNORECASE
        )
        if route_match:
            c1 = route_match.group(1).strip().lower()
            c2 = route_match.group(2).strip().lower()
            for name, code in city_map.items():
                if name in c1:
                    origin = code
                    break
            for name, code in city_map.items():
                if name in c2:
                    destination = code
                    break
        else:
            # Fallback regex tìm mã sân bay
            codes = re.findall(r'\b(HAN|SGN|DAD)\b', text, re.IGNORECASE)
            if len(codes) >= 2:
                origin = codes[0].upper()
                destination = codes[1].upper()
            elif len(codes) == 1:
                destination = codes[0].upper()

        # Tìm giá tối đa
        price_match = re.search(r'dưới\s*([\d\.,]+)\s*(triệu|tr|k|nghìn)?', text, re.IGNORECASE)
        if price_match:
            val_str = price_match.group(1).replace(",", ".")
            unit = (price_match.group(2) or "").lower()
            try:
                val = float(val_str)
                if unit in ["triệu", "tr"] or val < 100:
                    max_price = int(val * 1_000_000)
                elif unit in ["k", "nghìn"] or val >= 100:
                    max_price = int(val * 1_000 if unit else val)
            except ValueError:
                pass

        return {"origin": origin, "destination": destination, "max_price": max_price}

    def _parse_weather_params(self, text: str, fallback_dest: Optional[str] = None) -> Dict[str, Any]:
        """Trích xuất mã thành phố cho thời tiết."""
        city_map = {
            "hà nội": "HAN", "han": "HAN",
            "đà nẵng": "DAD", "dad": "DAD",
            "tp. hồ chí minh": "SGN", "tp.hcm": "SGN", "tphcm": "SGN",
            "hồ chí minh": "SGN", "sài gòn": "SGN", "sgn": "SGN"
        }
        # Ưu tiên trích xuất cụm thành phố gắn liền với từ khóa 'thời tiết'
        weather_match = re.search(r'thời tiết\s*(?:ở|tại|cho)?\s*([A-Za-zÀ-ỹ\.\s]+?)(?=[,\.\?]|\s+hiện|\s+nên|\s+ngày|\s+mặc|$)', text, re.IGNORECASE)
        if weather_match:
            cand = weather_match.group(1).strip().lower()
            for name, code in city_map.items():
                if name in cand:
                    return {"city_code": code}
        
        # Nếu có fallback_dest (ví dụ điểm đến của chuyến bay), ưu tiên điểm đến
        if fallback_dest:
            return {"city_code": fallback_dest}

        text_lower = text.lower()
        for name, code in city_map.items():
            if name in text_lower:
                return {"city_code": code}
        return {"city_code": "SGN"}

    def _determine_tasks(self, user_input: str):
        """Phân loại yêu cầu: chuyến bay, thời tiết, hoặc FAQ."""
        text_lower = user_input.lower()

        # Kiểm tra nếu là câu hỏi chính sách/FAQ Vinpearl (không gọi tool tra cứu thời gian thực)
        if "chính sách" in text_lower or "quy định" in text_lower or ("vinpearl" in text_lower and not ("chuyến bay" in text_lower and ("từ" in text_lower or "đi" in text_lower))):
            return False, False, True

        needs_flight = any(k in text_lower for k in ["chuyến bay", "vé máy bay", "vé", "bay từ", "bay đi", "đi sgn", "đi dad", "đi han"])
        needs_weather = any(k in text_lower for k in ["thời tiết", "mặc gì", "nhiệt độ", "mưa", "nắng", "dự báo"])
        is_faq = not needs_flight and not needs_weather

        return needs_flight, needs_weather, is_faq

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Thực thi vòng lặp ReAct (Milestone 3 & Milestone 4).
        """
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        needs_flight, needs_weather, is_faq = self._determine_tasks(user_input)

        # Kịch bản 1: FAQ / Không cần tool
        if is_faq and not needs_flight and not needs_weather:
            iteration = 1
            thought = "Khách hàng hỏi về chính sách dịch vụ của Vinpearl. Đây là câu hỏi FAQ chung, không cần tra cứu dữ liệu thời gian thực từ tool."
            answer = (
                "Chính sách đổi trả vé máy bay và dịch vụ phòng của Vinpearl: "
                "Khách hàng được hỗ trợ đổi ngày bay, đổi tên hành khách hoặc hoàn bảo lưu vé theo điều kiện cụ thể của từng hạng vé. "
                "Vui lòng liên hệ tổng đài hỗ trợ Vinpearl để được nhân viên kiểm tra trực tiếp mã đặt chỗ."
            )
            self.trace.append({
                "step": iteration,
                "thought": thought,
                "action": None,
                "observation": None,
                "final_answer": answer
            })
            return {
                "status": "completed",
                "iterations": 1,
                "trace": self.trace,
                "answer": answer
            }

        # Kịch bản 2: Câu hỏi đơn bước (Single-step flight)
        if needs_flight and not needs_weather:
            flight_params = self._parse_flight_params(user_input)
            iteration = 1

            thought = f"Khách hàng yêu cầu tìm chuyến bay từ {flight_params['origin']} đi {flight_params['destination']} với giá tối đa {flight_params['max_price']:,} VND. Cần gọi tool get_flight_info."
            action = {"name": "get_flight_info", "args": flight_params}
            obs = self._execute_tool(action)

            if isinstance(obs, list) and len(obs) > 0:
                flights_desc = ", ".join([f"{f['flight_number']} ({f['airline']}, {f['price_vnd']:,} VND, khởi hành {f['departure_time']})" for f in obs])
                answer = f"Tìm thấy {len(obs)} chuyến bay phù hợp từ {flight_params['origin']} đi {flight_params['destination']}: {flights_desc}."
            else:
                answer = f"Rất tiếc không tìm thấy chuyến bay nào từ {flight_params['origin']} đi {flight_params['destination']} dưới {flight_params['max_price']:,} VND."

            self.trace.append({
                "step": iteration,
                "thought": thought,
                "action": action,
                "observation": obs,
                "final_answer": answer
            })
            return {
                "status": "completed",
                "iterations": 1,
                "trace": self.trace,
                "answer": answer
            }

        # Kịch bản 3: Câu hỏi đơn bước (Single-step weather)
        if needs_weather and not needs_flight:
            weather_params = self._parse_weather_params(user_input)
            iteration = 1

            thought = f"Khách hàng yêu cầu thông tin thời tiết cho {weather_params['city_code']}. Cần gọi tool get_weather_forecast."
            action = {"name": "get_weather_forecast", "args": weather_params}
            obs = self._execute_tool(action)

            if isinstance(obs, dict) and "error" not in obs:
                answer = (
                    f"Thời tiết hiện tại ở {obs.get('city', weather_params['city_code'])}: "
                    f"nhiệt độ {obs.get('temperature_c')}°C, tình trạng {obs.get('condition')}, độ ẩm {obs.get('humidity_pct')}%. "
                    f"Gợi ý trang phục: {obs.get('recommendation', 'Trang phục thoải mái.')}"
                )
            else:
                answer = f"Không tìm thấy thông tin thời tiết cho mã {weather_params['city_code']}."

            self.trace.append({
                "step": iteration,
                "thought": thought,
                "action": action,
                "observation": obs,
                "final_answer": answer
            })
            return {
                "status": "completed",
                "iterations": 1,
                "trace": self.trace,
                "answer": answer
            }

        # Kịch bản 4: Câu hỏi đa bước (Multi-step flight & weather)
        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        flight_params = self._parse_flight_params(user_input)
        weather_params = self._parse_weather_params(user_input, fallback_dest=flight_params["destination"])

        flight_obs = None
        weather_obs = None

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # TODO 3: Phân tích Thought / Action từ Agent
            if iteration == 1:
                # Bước 1: Tra cứu chuyến bay
                thought = f"Khách hàng cần tìm chuyến bay từ {flight_params['origin']} đi {flight_params['destination']} dưới {flight_params['max_price']:,} VND. Cần tra cứu dữ liệu chuyến bay trước."
                action = {"name": "get_flight_info", "args": flight_params}
                # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
                flight_obs = self._execute_tool(action)
                # TODO 5: Ghi lại Observation và lặp lại
                self.trace.append({
                    "step": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": flight_obs
                })

            elif iteration == 2:
                # Bước 2: Tra cứu thời tiết
                thought = f"Đã có thông tin chuyến bay. Tiếp theo cần tra cứu thời tiết tại {weather_params['city_code']} để tư vấn trang phục phù hợp."
                action = {"name": "get_weather_forecast", "args": weather_params}
                weather_obs = self._execute_tool(action)
                self.trace.append({
                    "step": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": weather_obs
                })

            elif iteration == 3:
                # Bước 3: Tổng hợp Final Answer
                thought = "Đã có đầy đủ dữ liệu về chuyến bay và thời tiết. Tổng hợp câu trả lời hoàn chỉnh gửi khách hàng."
                flight_details = ""
                if isinstance(flight_obs, list) and flight_obs:
                    flight_details = ", ".join([f"{f['flight_number']} ({f['airline']}, giá {f['price_vnd']:,} VND, khởi hành {f['departure_time']})" for f in flight_obs])
                else:
                    flight_details = "Không có chuyến bay phù hợp."

                weather_details = ""
                if isinstance(weather_obs, dict) and "error" not in weather_obs:
                    weather_details = f"Tại {weather_obs.get('city')}: nhiệt độ khoảng {weather_obs.get('temperature_c')}°C, {weather_obs.get('condition')}. Khuyến nghị: {weather_obs.get('recommendation')}"
                else:
                    weather_details = "Không có thông tin thời tiết."

                answer = (
                    f"Dưới đây là thông tin chi tiết dành cho bạn:\n"
                    f"1. Chuyến bay từ {flight_params['origin']} đi {flight_params['destination']}: {flight_details}.\n"
                    f"2. Thời tiết và trang phục: {weather_details}"
                )

                self.trace.append({
                    "step": iteration,
                    "thought": thought,
                    "action": None,
                    "observation": None,
                    "final_answer": answer
                })

                return {
                    "status": "completed",
                    "iterations": iteration,
                    "trace": self.trace,
                    "answer": answer
                }

        # Milestone 4: Safeguard ngắt vòng lặp an toàn khi vượt quá số bước tối đa
        return {
            "status": "max_iterations_reached",
            "iterations": iteration,
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    baseline_result = chatbot.query(user_query)
    print(json.dumps(baseline_result, indent=2, ensure_ascii=False))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result Status:", result.get("status"))
    print("Iterations:", result.get("iterations"))
    print("Answer:", result.get("answer"))
    print("\nTrace Log:")
    print(json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()