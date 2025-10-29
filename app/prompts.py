ANSWER_PROMPT = (
"Bạn là trợ lý tiếng Việt, trả lời ngắn gọn, chính xác.\n"
"Câu hỏi người dùng: {question}\n"
"Câu trả lời đề xuất từ cơ sở tri thức: {candidate}\n\n"
"Hãy diễn đạt lại câu trả lời cuối cùng bằng tiếng Việt rõ ràng, nếu không đủ thông tin thì nói: \"Không đủ thông tin để trả lời.\""
)


CRITIQUE_PROMPT = (
"Bạn là người kiểm chứng.\n"
"- Kiểm tra xem câu trả lời đề xuất có trả lời đúng câu hỏi hay không.\n"
"- Nếu có điểm mơ hồ/không đúng, hãy chỉ ra ngắn gọn.\n"
"- Viết \"Đáp án cuối\" bằng 1 câu ngắn gọn. Nếu không đủ bằng chứng, trả lời: \"Không đủ thông tin để trả lời.\"\n\n"
"Câu hỏi: {question}\n"
"Câu trả lời đề xuất: {candidate}\n\n"
"Phản biện:\n"
"Đáp án cuối:"
)