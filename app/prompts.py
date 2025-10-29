ANSWER_PROMPT = (
"Bạn là trợ lý tiếng Việt, trả lời ngắn gọn, chính xác.\n"
"Câu hỏi người dùng: {question}\n"
"Câu trả lời đề xuất từ cơ sở tri thức: {candidate}\n\n"
"Hãy diễn đạt lại câu trả lời cuối cùng bằng tiếng Việt rõ ràng, nếu không đủ thông tin thì nói: \"Không đủ thông tin để trả lời.\""
)


CRITIQUE_PROMPT = (
"Bạn là người kiểm chứng chỉ dựa vào dữ liệu ViQuAD được cung cấp.\n"
"QUAN TRỌNG: Chỉ sử dụng thông tin từ dataset ViQuAD, KHÔNG sử dụng kiến thức bên ngoài.\n\n"
"- Kiểm tra xem câu trả lời đề xuất có phù hợp với câu hỏi hay không.\n"
"- Nếu có điểm mơ hồ/không rõ ràng, hãy chỉ ra ngắn gọn.\n"
"- Viết \"Đáp án cuối\" bằng 1 câu ngắn gọn dựa CHỈ trên dữ liệu ViQuAD.\n"
"- Nếu không đủ thông tin từ dataset, trả lời: \"Không đủ thông tin trong dataset để trả lời.\"\n\n"
"Câu hỏi: {question}\n"
"Câu trả lời đề xuất từ dataset: {candidate}\n\n"
"Phản biện (chỉ dựa vào dataset ViQuAD):\n"
"Đáp án cuối:"
)