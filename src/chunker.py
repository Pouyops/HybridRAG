from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

from config import settings

_HEADER_LEVELS = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


class Chunker():
    def __init__(
        self,
        embedding_fn,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        markdown_max_tokens=settings.markdown_max_tokens,
    ):
        self.embedding_fn = embedding_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.markdown_max_tokens = markdown_max_tokens

    def chunk_documents(self, text, metadata, strategy="TokenRecursive"):
        if strategy == "Markdown":
            return self._markdown_chunking(text, metadata)
        elif strategy == "Semantic":
            return self._semantic_chunking(text, metadata)
        else:
            return self._token_recursive_chunking(text, metadata)

    def _markdown_chunking(self, text, metadata):
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADER_LEVELS,
            strip_headers=False
        )
        sections = markdown_splitter.split_text(text)
        chunks = []
        for section in sections:
            chunks.extend(self._cap_section(section))
        for chunk in chunks:
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_strategy"] = "MarkdownHeader"
        return chunks

    def _cap_section(self, section):
        """Split a header-delimited section that exceeds markdown_max_tokens.

        Uncapped, a long section (RFC 9110's "9.3. Method Definitions" is
        ~5k tokens) becomes one chunk: its embedding is diluted, the
        cross-encoder only sees the start of it, and it floods the prompt.
        Every piece after the first is prefixed with the section's heading
        path, so a continuation chunk still says which section it is from.
        """
        if not self.markdown_max_tokens:
            return [section]
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.markdown_max_tokens,
            chunk_overlap=self.chunk_overlap,
        )
        pieces = splitter.split_text(section.page_content)
        if len(pieces) <= 1:
            return [section]
        heading_path = "\n".join(
            f"{marker} {section.metadata[key]}"
            for marker, key in _HEADER_LEVELS
            if key in section.metadata
        )
        capped = []
        for i, piece in enumerate(pieces):
            if i > 0 and heading_path:
                piece = f"{heading_path} (continued)\n\n{piece}"
            capped.append(Document(page_content=piece, metadata=dict(section.metadata)))
        return capped

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
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = token_splitter.create_documents([text])
        for chunk in chunks:
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_strategy"] = "TokenRecursive"
        return chunks