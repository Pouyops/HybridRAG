"""Build a persisted Chroma index from a data directory.

The Chroma/BM25 index directories (chroma_main_db/, chroma_db_*/) are
generated artifacts and are gitignored — this script is what reproduces them,
instead of committing the binary DB blobs themselves.

Usage:
    python scripts/build_index.py --data-dir ./data --persist-dir ./chroma_main_db --strategy TokenRecursive
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from config import settings
from src.chunker import Chunker
from src.indexer import indexer
from src.loader import multiloader


def build_index(data_dir: str, persist_dir: str, strategy: str, api_key: str) -> int:
    loader = multiloader(data_dir)
    documents = loader._document_loader()
    if not documents:
        raise SystemExit(f"No documents found in {data_dir}. Run `python scripts/fetch_corpus.py` to download the default corpus.")

    embedding_fn = OpenAIEmbeddings(model=settings.embedding_model)
    chunker = Chunker(embedding_fn=embedding_fn)
    chunks = []
    for doc in documents:
        chunks.extend(
            chunker.chunk_documents(doc.page_content, doc.metadata, strategy=strategy)
        )

    idx = indexer(api_key=api_key)
    idx.create_indexes(chunks, persist_directory=persist_dir)

    print(
        f"Indexed {len(chunks)} chunks from {len(documents)} documents "
        f"({strategy}) into {persist_dir}"
    )
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a persisted Chroma index from a data directory."
    )
    parser.add_argument(
        "--data-dir", default="./data/", help="Directory containing documents to index."
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_main_db",
        help="Directory to persist the Chroma index in.",
    )
    parser.add_argument(
        "--strategy",
        default="TokenRecursive",
        choices=["TokenRecursive", "Markdown", "Semantic"],
        help="Chunking strategy to use.",
    )
    args = parser.parse_args()

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY environment variable not set.")

    build_index(args.data_dir, args.persist_dir, args.strategy, OPENAI_API_KEY)
