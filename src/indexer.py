from collections import defaultdict

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

from config import settings


class indexer:
    def __init__(self, api_key=None, model_name=settings.embedding_model):
        # OpenAIEmbeddings reads OPENAI_API_KEY from the environment
        # automatically (populated by load_dotenv() in the entry point).
        self.embeddings = OpenAIEmbeddings(model=model_name)

    def _prepare_metadata(self, chunks):
        file_chunk_counters = defaultdict(int)
        processed_chunks = []

        for chunk in chunks:
            source_file = chunk.metadata.get("filepath", "unknown")
            index = file_chunk_counters[source_file]
            file_chunk_counters[source_file] += 1

            heading = "N/A"
            for header_key in ["Header 3", "Header 2", "Header 1"]:
                if header_key in chunk.metadata:
                    heading = chunk.metadata[header_key]
                    break

            chunk.metadata["chunk_index"] = index
            chunk.metadata["section_heading"] = heading
            chunk.metadata["character_count"] = len(chunk.page_content)

            processed_chunks.append(chunk)

        return processed_chunks

    def create_indexes(self, chunks, persist_directory="./chroma_db"):
        processed_chunks = self._prepare_metadata(chunks)

        # Chroma.from_documents() appends to whatever collection already lives
        # in persist_directory, so rebuilding into the same directory (every
        # app/Streamlit/main.py restart) would duplicate the whole corpus.
        # Start from an empty collection so the index mirrors `chunks` exactly.
        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        vectorstore.reset_collection()
        vectorstore.add_documents(processed_chunks)
        bm25_retriever = BM25Retriever.from_documents(
            processed_chunks, k=settings.retrieval_depth
        )

        return vectorstore, bm25_retriever
