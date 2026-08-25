# Evidence — Day 22: LangSmith + Prompt Versioning

Bằng chứng thực thi cho 4 nhiệm vụ của lab. Mỗi file tương ứng một mục trong rubric chấm điểm.

## Danh sách bằng chứng

| File | Nội dung | Nhiệm vụ |
|---|---|---|
| `01_langsmith_traces.png` | Giao diện LangSmith hiển thị ≥ 50 traces (câu hỏi, context truy xuất, câu trả lời) | 1 |
| `02_prompt_hub.png` | Prompt Hub với 2 prompt đã đặt tên: `my-rag-prompt-v1`, `my-rag-prompt-v2` | 2 |
| `02_ab_routing_log.txt` | Console log A/B routing — 50 câu hỏi, mỗi câu gắn nhãn `[prompt-v1]`/`[prompt-v2]` theo MD5 hash của request_id (tất định) | 2 |
| `03_ragas_scores.png` | Terminal output bảng so sánh điểm RAGAS V1 vs V2 | 3 |
| `03_ragas_report.json` | Báo cáo JSON chứa điểm 4 metrics của cả 2 phiên bản | 3 |
| `04_pii_demo_log.txt` | Console log 6 test case PII — email, phone, SSN, credit card bị redact bằng regex | 4 |
| `04_json_demo_log.txt` | Console log 5 test case JSON — fences/single quotes/trailing comma được tự sửa; JSON hỏng hoàn toàn trả về fallback | 4 |

## Cấu hình chạy

| Thành phần | Provider / Model | Ghi chú |
|---|---|---|
| LLM | OpenRouter `stealth/ox-alpha` | Miễn phí ($0/$0), client retry 30 lần để hấp thụ 429 từ shared pool |
| Embeddings | Ollama `nomic-embed-text` (local) | Tách provider qua biến `EMBEDDING_PROVIDER` trong `.env` |
| Vectorstore | FAISS | 107 chunks, k=3 |
| Tracing | LangSmith `day22-lab` | `LANGCHAIN_TRACING_V2=true` |

## Phân tích kết quả V1 vs V2

> ⏳ **Chờ điền sau khi Bước 3 (RAGAS) hoàn thành.**

### Điểm số

| Metric | V1 | V2 |
|---|---|---|
| faithfulness | TBD | TBD |
| answer_relevancy | TBD | TBD |
| context_recall | TBD | TBD |
| context_precision | TBD | TBD |

### Phân tích

> TODO: điền 2–4 đoạn ngắn:
> - Phiên bản nào thắng tổng thể và vì sao
> - Sự khác biệt giữa 2 system prompt (V1 trợ lý ngắn gọn 2–4 câu vs V2 chuyên gia hàn lâm 3–5 câu) tác động thế nào lên từng metric
> - Faithfulness có đạt mục tiêu ≥ 0.8 không

### Quan sát về multilingual evaluation

> TODO (nếu answer_relevancy thấp bất thường): hệ QA pairs và knowledge base là tiếng Anh,
> nhưng cả 2 system prompt viết bằng tiếng Việt và không khóa ngôn ngữ đầu ra.
> Metric `answer_relevancy` nhúng câu trả lời bằng `nomic-embed-text` (English-centric)
> nên nếu model trả lời tiếng Việt, điểm bị kéo xuống giả tạo dù nội dung đúng.

## Ghi chú kỹ thuật

- **A/B routing tất định**: `MD5(request_id)` → hash chẵn chọn V1, lẻ chọn V2. Cùng request_id luôn cho cùng phiên bản.
- **Guardrails**: `on_fail=OnFailAction.FIX` truyền vào constructor validator (không phải `Guard.use()`); redact PII bằng regex 4 pattern; JSON repair gồm strip fences + đổi quotes + xóa trailing commas, fallback `{"error": "Unparseable JSON"}` khi không sửa được.
- **Checkpoint**: Bước 3 lưu `data/rag_outputs_{v}.json` sau generate và `data/ragas_scores_{v}.json` sau eval từng version — chạy lại không mất công tính toán.
