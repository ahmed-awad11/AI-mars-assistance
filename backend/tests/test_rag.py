"""
Marswalk AI Assistant - RAG/Knowledge Base Tests
"""
import pytest
import tempfile
import os
from pathlib import Path
from backend.rag import KnowledgeBase
from backend.config import settings


class TestKnowledgeBase:
    """Test knowledge base functionality."""

    @pytest.fixture
    async def kb(self):
        """Create a temporary knowledge base for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override settings for test
            original_chroma = settings.CHROMA_PERSIST_DIR
            original_docs = settings.DOCUMENTS_DIR
            settings.CHROMA_PERSIST_DIR = os.path.join(tmpdir, 'chroma')
            settings.DOCUMENTS_DIR = os.path.join(tmpdir, 'documents')
            os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
            os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)

            kb = KnowledgeBase()
            await kb.initialize()
            yield kb

            # Restore settings
            settings.CHROMA_PERSIST_DIR = original_chroma
            settings.DOCUMENTS_DIR = original_docs

    @pytest.mark.asyncio
    async def test_initialize(self, kb):
        """Test knowledge base initialization."""
        assert kb._initialized
        assert kb.embedding_model is not None
        assert kb.chroma_client is not None
        assert kb.collection is not None

    @pytest.mark.asyncio
    async def test_chunk_text(self, kb):
        """Test text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        chunks = kb._chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 120  # chunk_size + overlap

    @pytest.mark.asyncio
    async def test_get_embedding(self, kb):
        """Test embedding generation."""
        embedding = kb._get_embedding("Test text for embedding")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_ingest_text_file(self, kb):
        """Test ingesting a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Mars is the fourth planet from the Sun. It has a thin atmosphere.")
            f.flush()
            filepath = f.name

        try:
            doc = await kb.ingest_document(filepath, title="Mars Facts", source_type="txt")
            assert doc.title == "Mars Facts"
            assert doc.source_type == "txt"
            assert doc.chunk_count > 0
            assert doc.is_processed
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_ingest_markdown_file(self, kb):
        """Test ingesting a markdown file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Jezero Crater\n\nJezero is a crater on Mars.\n\n## Delta\nThe delta is well preserved.")
            f.flush()
            filepath = f.name

        try:
            doc = await kb.ingest_document(filepath, source_type="md")
            assert doc.source_type == "md"
            assert doc.chunk_count > 0
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_search(self, kb):
        """Test knowledge base search."""
        # Ingest test documents
        docs = [
            ("Jezero Crater is located at 18.4°N, 77.5°E. It contains a well-preserved delta.", "jezero.txt"),
            ("Perseverance rover landed in Jezero on February 18, 2021.", "perseverance.txt"),
            ("The Martian atmosphere is 95% carbon dioxide.", "mars_atmosphere.txt"),
        ]

        for content, filename in docs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                f.flush()
                await kb.ingest_document(f.name, title=filename.replace('.txt', ''))
                os.unlink(f.name)

        # Search
        results = await kb.search("Jezero delta", top_k=2)
        assert len(results) > 0
        assert any("delta" in r["content"].lower() for r in results)

        results = await kb.search("Perseverance landing", top_k=2)
        assert len(results) > 0
        assert any("2021" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, kb):
        """Test search with metadata filter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document for filtering")
            f.flush()
            doc = await kb.ingest_document(f.name, metadata={"mission": "test"})
            os.unlink(f.name)

        results = await kb.search("filtering", filter={"document_id": doc.id})
        assert len(results) > 0
        assert results[0]["metadata"]["document_id"] == doc.id

    @pytest.mark.asyncio
    async def test_list_documents(self, kb):
        """Test listing documents."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Document one")
            f.flush()
            await kb.ingest_document(f.name, title="Doc One")
            os.unlink(f.name)

        docs = await kb.list_documents()
        assert len(docs) >= 1
        assert any(d.title == "Doc One" for d in docs)

    @pytest.mark.asyncio
    async def test_delete_document(self, kb):
        """Test document deletion."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Document to delete")
            f.flush()
            doc = await kb.ingest_document(f.name, title="To Delete")
            os.unlink(f.name)

        # Verify it exists
        docs_before = await kb.list_documents()
        assert any(d.id == doc.id for d in docs_before)

        # Delete
        success = await kb.delete_document(doc.id)
        assert success

        # Verify it's gone
        docs_after = await kb.list_documents()
        assert not any(d.id == doc.id for d in docs_after)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])