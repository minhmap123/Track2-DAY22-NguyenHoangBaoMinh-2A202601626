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

### Điểm số

| Metric             |     V1 |     V2 | Thắng |
|---|---|---|---|
| faithfulness       | **0.9701** ⭐ | 0.8389 | V1 |
| answer_relevancy   | **0.8940** | 0.8181 | V1 |
| context_recall     | 0.9787 | 0.9767 | ≈ ngang |
| context_precision  | 0.9244 | 0.9250 | ≈ ngang |

**Mục tiêu faithfulness ≥ 0.8: ĐẠT** (cả 2 version đều vượt; V1 đạt thêm ngưỡng 0.9).

### Phân tích

**V1 thắng trên cả 2 metric phụ thuộc vào chất lượng sinh văn bản** (faithfulness, answer_relevancy), trong khi 2 metric thuần retrieval (context_recall, context_precision) gần như trùng khớp giữa 2 version. Điều này nhất quán với thiết kế thí nghiệm: cả 2 prompt dùng chung retriever (FAISS, k=3) nên phần contexts hoàn toàn giống nhau — khác biệt chỉ nằm ở cách LLM diễn đạt câu trả lời.

**Vì sao V1 (ngắn gọn 2–4 câu) ăn điểm faithfulness cao hơn V2 (hàn lâm 3–5 câu)?** Faithfulness đo tỷ lệ claim trong câu trả lời có căn cứ trong contexts. Câu trả lời càng dài và "hàn lâm", model càng có xu hướng bổ sung kiến thức nền (elaboration) không có trong context — mỗi câu thêm là một cơ hội phát sinh unsupported claim. Prompt V1 ép câu trả lời trực diện nên gần như mọi câu đều truy vết được về nguồn. Đây là minh họa kinh điển cho trade-off: prompt giàu tính hàn lâm tăng độ phong phú nhưng giảm grounding.

**answer_relevancy thấp hơn faithfulness ở cả 2 version** là pattern thường thấy của RAGAS: metric này sinh câu hỏi ngược từ câu trả lời rồi so độ tương đồng embedding với câu hỏi gốc — câu trả lời dài (V2) dễ lan man khỏi trọng tâm câu hỏi, khiến V2 tụt nhiều hơn (0.818 vs 0.894).

### Quan sát về multilingual evaluation

Hệ QA pairs và knowledge base là tiếng Anh, nhưng cả 2 system prompt viết bằng tiếng Việt và không khóa ngôn ngữ đầu ra. Rủi ro lý thuyết: `answer_relevancy` nhúng câu trả lời bằng `nomic-embed-text` (English-centric), nếu model trả lời tiếng Việt thì điểm bị kéo xuống giả tạo. Trong thực tế chạy, điểm answer_relevancy ở mức healthy (0.82–0.89) cho thấy ox-alpha đã tự động trả lời tiếng Anh theo ngữ cảnh câu hỏi — language leakage xảy ra nhưng hạn chế. Nếu muốn loại trừ hẳn biến nhiễu này, có thể thêm dòng `"Always answer in English."` vào system prompt.

### Ghi chú về độ tin cậy của phép đo

Eval chia theo chunk 10 samples × 4 metrics với resume-checkpoint; tỷ lệ sample lỗi parser (ox-alpha trả sai JSON format cho judge, bị gán NaN rồi nanmean loại bỏ) khoảng 10–15% mỗi chunk. Mean tính trên ~85–90% samples — đủ đại diện cho so sánh tương đối giữa 2 version.

## Ghi chú kỹ thuật

- **A/B routing tất định**: `MD5(request_id)` → hash chẵn chọn V1, lẻ chọn V2. Cùng request_id luôn cho cùng phiên bản.
- **Guardrails**: `on_fail=OnFailAction.FIX` truyền vào constructor validator (không phải `Guard.use()`); redact PII bằng regex 4 pattern; JSON repair gồm strip fences + đổi quotes + xóa trailing commas, fallback `{"error": "Unparseable JSON"}` khi không sửa được.
- **Checkpoint**: Bước 3 lưu `data/rag_outputs_{v}.json` sau generate và `data/ragas_scores_{v}.json` sau eval từng version — chạy lại không mất công tính toán.
