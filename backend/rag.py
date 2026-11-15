"""
Marswalk AI Assistant - RAG Knowledge System
Document ingestion and retrieval with optional ChromaDB/sentence-transformers support.
Falls back to SQLite-based text search when dependencies are unavailable.
"""
import hashlib
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from config import settings
from models import Document, DocumentChunk, SessionLocal

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available - using SQLite fallback for knowledge base")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - using text search fallback")


class KnowledgeBase:
    """RAG knowledge base for Marswalk scientific documents."""

    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False
        self._use_fallback = not (CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE)

    async def initialize(self):
        """Initialize embedding model and ChromaDB, or fallback to SQLite."""
        if self._initialized:
            return

        if self._use_fallback:
            logger.info("Knowledge base initialized in fallback mode (SQLite text search)")
            self._initialized = True
            return

        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            )

            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            self._initialized = True
            logger.info("Knowledge base initialized successfully with ChromaDB")

        except Exception as e:
            logger.warning(f"ChromaDB initialization failed, falling back to SQLite: {e}")
            self._use_fallback = True
            self._initialized = True

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self._use_fallback:
            # Return a simple hash-based vector for fallback
            return self._text_to_vector(text)
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized")
        return self.embedding_model.encode(text).tolist()

    def _text_to_vector(self, text: str) -> List[float]:
        """Simple text-to-vector for fallback (hash-based)."""
        # Create a simple deterministic vector from text
        words = re.findall(r'\w+', text.lower())
        vector = [0.0] * 384  # Standard embedding size
        for i, word in enumerate(words[:384]):
            vector[i] = hash(word) % 1000 / 1000.0
        return vector

    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size * 0.5:
                    end = start + break_point + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]

    def _simple_search(self, query: str, top_k: int = 5, filter: Dict = None) -> List[Dict[str, Any]]:
        """Simple text search in SQLite (fallback)."""
        db = SessionLocal()
        try:
            query_terms = query.lower().split()
            chunks = db.query(DocumentChunk).join(Document).all()

            results = []
            for chunk in chunks:
                # Apply filter
                chunk_meta = chunk.metadata or {}
                if filter:
                    if filter.get("source_type") and chunk_meta.get("source_type") != filter["source_type"]:
                        continue
                    if filter.get("document_id") and chunk_meta.get("document_id") != filter["document_id"]:
                        continue

                # Simple text matching score
                content_lower = chunk.content.lower()
                score = sum(1 for term in query_terms if term in content_lower)
                if score > 0:
                    # Normalize by content length
                    relevance = min(score / len(query_terms), 1.0)
                    results.append({
                        "id": chunk.embedding_id or f"chunk_{chunk.id}",
                        "content": chunk.content[:500],
                        "metadata": chunk_meta,
                        "distance": 1.0 - relevance,
                        "relevance": relevance
                    })

            # Sort by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:top_k]

        finally:
            db.close()

    async def ingest_document(
        self,
        filepath: str,
        title: str = None,
        source_type: str = None,
        metadata: Dict = None
    ) -> Document:
        """Ingest a document into the knowledge base."""
        await self.initialize()

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read file content
        content = self._read_file(path)
        if not content:
            raise ValueError("Empty file content")

        # Compute content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check if already ingested
        db = SessionLocal()
        try:
            existing = db.query(Document).filter(Document.content_hash == content_hash).first()
            if existing:
                logger.info(f"Document already ingested: {existing.filename}")
                return existing

            # Determine source type
            if source_type is None:
                source_type = path.suffix.lower().lstrip('.')
                if source_type not in ['pdf', 'txt', 'md', 'json']:
                    source_type = 'txt'

            # Create document record
            doc = Document(
                filename=path.name,
                filepath=str(path.absolute()),
                content_hash=content_hash,
                title=title or path.stem,
                source_type=source_type,
                metadata=metadata or {},
                chunk_count=0,
                is_processed=False
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # Chunk text
            chunks = self._chunk_text(content)
            doc.chunk_count = len(chunks)

            if not self._use_fallback and self.collection:
                # Add to ChromaDB
                chunk_ids = []
                chunk_embeddings = []
                metadatas = []
                chunk_documents = []

                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc.id}_chunk_{i}"
                    chunk_ids.append(chunk_id)

                    embedding = self._get_embedding(chunk_text)
                    chunk_embeddings.append(embedding)

                    chunk_meta = {
                        "document_id": doc.id,
                        "chunk_index": i,
                        "filename": doc.filename,
                        "title": doc.title,
                        "source_type": doc.source_type,
                        **(metadata or {})
                    }
                    metadatas.append(chunk_meta)
                    chunk_documents.append(chunk_text)

                    # Save chunk to SQL
                    chunk = DocumentChunk(
                        document_id=doc.id,
                        chunk_index=i,
                        content=chunk_text,
                        embedding_id=chunk_id,
                        metadata=chunk_meta
                    )
                    db.add(chunk)

                # Batch add to ChromaDB
                try:
                    self.collection.add(
                        ids=chunk_ids,
                        embeddings=chunk_embeddings,
                        metadatas=metadatas,
                        documents=chunk_documents
                    )
                except Exception as e:
                    logger.warning(f"ChromaDB add failed, using SQLite only: {e}")
                    self._use_fallback = True
            else:
                # Save chunks to SQLite only
                for i, chunk_text in enumerate(chunks):
                    chunk = DocumentChunk(
                        document_id=doc.id,
                        chunk_index=i,
                        content=chunk_text,
                        embedding_id=f"{doc.id}_chunk_{i}",
                        metadata={
                            "document_id": doc.id,
                            "chunk_index": i,
                            "filename": doc.filename,
                            "title": doc.title,
                            "source_type": doc.source_type,
                            **(metadata or {})
                        }
                    )
                    db.add(chunk)

            doc.is_processed = True
            db.commit()
            db.refresh(doc)

            logger.info(f"Ingested document: {doc.filename} ({len(chunks)} chunks) [{'ChromaDB' if not self._use_fallback else 'SQLite fallback'}]")
            return doc

        finally:
            db.close()

    def _read_file(self, path: Path) -> str:
        """Read file content based on extension."""
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            return self._read_pdf(path)
        elif suffix in ['.txt', '.md', '.json']:
            return path.read_text(encoding='utf-8')
        else:
            # Try as text
            try:
                return path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                raise ValueError(f"Cannot read file as text: {path}")

    def _read_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
        except ImportError:
            logger.warning("pypdf not installed, cannot read PDF")
            return ""
        except Exception as e:
            logger.error(f"PDF reading error: {e}")
            return ""

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict = None
    ) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant documents."""
        await self.initialize()

        if self._use_fallback:
            return self._simple_search(query, top_k, filter)

        query_embedding = self._get_embedding(query)

        # Build where filter for ChromaDB
        where = {}
        if filter:
            if "source_type" in filter:
                where["source_type"] = filter["source_type"]
            if "document_id" in filter:
                where["document_id"] = filter["document_id"]

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where if where else None,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "relevance": 1 - results["distances"][0][i]
                    })
            return formatted

        except Exception as e:
            logger.warning(f"ChromaDB search failed, falling back to SQLite: {e}")
            self._use_fallback = True
            return self._simple_search(query, top_k, filter)

    async def get_document(self, document_id: int) -> Optional[Document]:
        """Get document by ID."""
        db = SessionLocal()
        try:
            return db.query(Document).filter(Document.id == document_id).first()
        finally:
            db.close()

    async def list_documents(self, limit: int = 50) -> List[Document]:
        """List ingested documents."""
        db = SessionLocal()
        try:
            return db.query(Document).order_by(Document.created_at.desc()).limit(limit).all()
        finally:
            db.close()

    async def delete_document(self, document_id: int) -> bool:
        """Delete a document and its chunks."""
        await self.initialize()

        db = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                return False

            # Delete from ChromaDB if available
            if not self._use_fallback and self.collection:
                chunk_ids = [f"{document_id}_chunk_{i}" for i in range(doc.chunk_count)]
                if chunk_ids:
                    try:
                        self.collection.delete(ids=chunk_ids)
                    except Exception:
                        pass

            # Delete from SQL (cascades to chunks)
            db.delete(doc)
            db.commit()

            logger.info(f"Deleted document: {doc.filename}")
            return True

        finally:
            db.close()

    async def ingest_directory(self, directory: str = None, pattern: str = "**/*") -> List[Document]:
        """Ingest all supported documents from a directory."""
        dir_path = Path(directory or settings.DOCUMENTS_DIR)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        supported_ext = {'.pdf', '.txt', '.md', '.json'}
        files = [f for f in dir_path.glob(pattern) if f.suffix.lower() in supported_ext]

        ingested = []
        for file_path in files:
            try:
                doc = await self.ingest_document(str(file_path))
                ingested.append(doc)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")

        return ingested


# Global knowledge base instance
knowledge_base = KnowledgeBase()