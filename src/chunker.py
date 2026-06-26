from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

class Chunker():
    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn

    def chunk_documents(self, text, metadata, strategy="TokenRecursive"):
        if strategy == "Markdown":
            return self._markdown_chunking(text, metadata)
        elif strategy == "Semantic":
            return self._semantic_chunking(text, metadata)
        else:
            return self._token_recursive_chunking(text, metadata)

    def _markdown_chunking(self, text, metadata):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        chunks = markdown_splitter.split_text(text)
        for chunk in chunks:
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_strategy"] = "MarkdownHeader"
        return chunks

    def _semantic_chunking(self, text, metadata):
        semantic_splitter = SemanticChunker(
            self.embedding_fn,
            breakpoint_threshold_type="percentile"
        )
        chunks = semantic_splitter.create_documents([text])
        for chunk in chunks:
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_strategy"] = "Semantic"
        return chunks

    def _token_recursive_chunking(self, text, metadata):
        token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=50
        )
        chunks = token_splitter.create_documents([text])
        for chunk in chunks:
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_strategy"] = "TokenRecursive"
        return chunks