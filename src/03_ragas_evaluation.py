"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import types
# BẢN VÁ LỖI THƯ VIỆN RAGAS VỚI LANGCHAIN 0.3+:
if 'langchain_community.chat_models' not in sys.modules:
    sys.modules['langchain_community.chat_models'] = types.ModuleType('langchain_community.chat_models')
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    mock_vertex = types.ModuleType('langchain_community.chat_models.vertexai')
    mock_vertex.ChatVertexAI = type('ChatVertexAI', (object,), {})
    sys.modules['langchain_community.chat_models.vertexai'] = mock_vertex

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# TODO: Copy SYSTEM_V1 và SYSTEM_V2 mà bạn đã viết ở file 02_prompt_hub_ab_routing.py
SYSTEM_V1 = """Bạn là trợ lý AI hữu ích. Chỉ dựa dẫm vô nội dung context sau để trả lời. 
Giữ câu trả lời ngắn gọn, trực diện, khoảng 2-4 câu nếu có thể.

Context: 
{context}"""
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = """Bạn là chuyên gia AI. Hãy đọc kỹ hệ thống context bên dưới, cẩn thận xác định các thông tin liên quan và facts chuẩn xác để phản hồi.
Viết câu trả lời mạch lạc, mang tính hàn lâm, sử dụng các thuật ngữ và trình bày có tổ chức trong khoảng 3-5 câu.

Context: 
{context}"""
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 2b. Checkpoint — lưu/tải kết quả RAG để không phải generate lại ──────
def save_rag_checkpoint(results: list, version: str):
    path = Path(__file__).parent.parent / "data" / f"rag_outputs_{version}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã checkpoint {len(results)} outputs của {version} vào {path.name}")


def load_rag_checkpoint(version: str) -> list | None:
    path = Path(__file__).parent.parent / "data" / f"rag_outputs_{version}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # Chỉ dùng checkpoint nếu đủ 50 câu (tránh dở dang từ lần chạy trước)
    if len(data) == len(QA_PAIRS):
        print(f"♻️  Tải lại {len(data)} outputs của {version} từ checkpoint {path.name}")
        return data
    return None


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    # TODO: Retrieve documents từ retriever
    docs = retriever.invoke(question)

    # TODO: Tạo contexts là danh sách page_content (KHÔNG ghép chuỗi ở đây)
    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]   # phải là list[str] !

    # TODO: Ghép contexts thành 1 string để truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    # TODO: Chạy chain (prompt | llm | StrOutputParser()).invoke(...)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    # TODO: Trả về dict với answer và contexts (list)
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        # TODO: Gọi run_rag() cho câu hỏi hiện tại
        out = run_rag(retriever, llm, prompt, qa["question"])

        # TODO: Append vào results dict với 4 keys
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],        # out["answer"]
            "contexts":  out["contexts"],      # out["contexts"] — phải là list[str] !
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    save_rag_checkpoint(results, prompt_version)
    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # TODO: Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],         # r["question"]
            response=r["answer"],             # r["answer"]
            retrieved_contexts=r["contexts"], # r["contexts"]
            reference=r["reference"],         # r["reference"]
        )
        for r in rag_results
    ]

    # TODO: Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def _eval_one_chunk(dataset, llm_eval, emb_eval, version: str, chunk_idx: int) -> dict:
    """
    Eval MỘT chunk (10 samples) với checkpoint theo chunk.

    evaluate() nguyên khối là điểm yếu single-point-of-failure: crash giữa
    chừng = mất toàn bộ. Chia chunk + lưu đĩa mỗi chunk → crash chỉ mất
    tối đa 1 chunk; chạy lại tự bỏ qua chunk đã xong (♻️).
    """
    from ragas.run_config import RunConfig

    ckpt_path = Path(__file__).parent.parent / "data" / f"ragas_chunk_{version}_{chunk_idx:02d}.json"
    if ckpt_path.exists():
        print(f"  ♻️  Chunk {chunk_idx:02d}: đã có checkpoint — bỏ qua")
        return json.loads(ckpt_path.read_text(encoding="utf-8"))

    # Gọi evaluate() đầy đủ 4 metrics cho riêng chunk này
    # (shared pool của model free hay 429 → retry "siêu to": 30 lần, max_wait 120s)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=4, max_retries=30, max_wait=120, timeout=300),
    )

    # Sample lỗi parser bị gán NaN → chuẩn hoá về None để JSON hợp lệ
    vals = {}
    for key in METRIC_KEYS:
        raw = result[key]
        vals[key] = [
            float(v) if (v is not None and v == v) else None   # v == v: False khi NaN
            for v in raw
        ]

    ckpt_path.write_text(json.dumps(vals, indent=2), encoding="utf-8")
    done = sum(len([x for x in vs if x is not None]) for vs in vals.values())
    print(f"  💾 Chunk {chunk_idx:02d}: eval xong ({done} điểm hợp lệ) — đã checkpoint")
    return vals


def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics, chia theo chunk 10 samples.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: tổng thời gian tương đương evaluate() nguyên khối nhưng chịu lỗi tốt hơn hẳn.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (theo chunk 10 samples)")

    # Evaluator dùng JSON mode (ox-alpha hỗ trợ response_format) → bớt lỗi
    # RagasOutputParserException do model bọc JSON trong markdown fences
    llm_eval = get_llm(temperature=0, json_mode=True)
    emb_eval = get_embeddings()

    # Chia rag_results thành các chunk 10 và eval lần lượt (có resume từng chunk)
    all_vals = []
    for chunk_idx, start in enumerate(range(0, len(rag_results), 10)):
        chunk_results = rag_results[start:start + 10]
        dataset       = build_ragas_dataset(chunk_results)
        all_vals.append(_eval_one_chunk(dataset, llm_eval, emb_eval, version, chunk_idx))

    # Gom điểm: mỗi chunk trả {metric: [10 giá trị hoặc None]} → gộp phẳng
    # ⚠️ nanmean: None/NaN (sample lỗi parser) bị loại, không nhiễm mean.
    scores = {}
    for key in METRIC_KEYS:
        flat = [v for chunk in all_vals for v in chunk.get(key, []) if v is not None]
        scores[key] = float(np.nanmean(flat)) if flat else 0.0

    # 💾 Checkpoint điểm tổng của version
    ckpt_path = Path(__file__).parent.parent / "data" / f"ragas_scores_{version}.json"
    ckpt_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu checkpoint điểm {version} vào {ckpt_path.name}")

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # TODO: Tạo vectorstore
    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2 (ưu tiên checkpoint nếu có)
    v1_results = load_rag_checkpoint("v1") or collect_rag_outputs(vectorstore, "v1")
    v2_results = load_rag_checkpoint("v2") or collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # TODO: Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    # TODO: Ghi report vào file bằng json.dumps hoặc json.dump
    # Gợi ý: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
