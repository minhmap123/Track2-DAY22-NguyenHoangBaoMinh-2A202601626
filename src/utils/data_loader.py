"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    from langchain_community.vectorstores import FAISS
    import time

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    # Lách giới hạn 100 request/phút của API Google miễn phí —
    # CHỈ cần khi embedding chạy bằng Gemini; Ollama (local) không bị limit này.
    emb_provider = (config.EMBEDDING_PROVIDER or config.PROVIDER or "").lower()
    if len(chunks) > 90 and emb_provider == "gemini":
        vectorstore = FAISS.from_texts(chunks[:90], embeddings)
        print("⏳ Hệ thống nghỉ 65 giây để vượt rào Rate Limit của cấp Free Tier từ Google...")
        time.sleep(65)
        vectorstore.add_texts(chunks[90:])
    else:
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
