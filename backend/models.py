"""
Marswalk AI Assistant - Database Models
SQLAlchemy models for conversation history and mission memory.
"""
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime,
    ForeignKey, JSON, Boolean, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.pool import StaticPool

from config import settings

Base = declarative_base()


class Conversation(Base):
    """Conversation sessions with the AI assistant."""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    title = Column(String(256), default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    mission_context = relationship("MissionContext", back_populates="conversation", uselist=False, cascade="all, delete-orphan")


class Message(Base):
    """Individual messages in a conversation."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=False)
    tool_calls = Column(JSON, nullable=True)  # Tool calls made by assistant
    tool_results = Column(JSON, nullable=True)  # Results from tool execution
    metadata = Column("metadata", JSON, nullable=True)  # Additional metadata (column name: metadata)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class MissionContext(Base):
    """Mission-specific context and memory."""
    __tablename__ = "mission_context"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), unique=True, nullable=False)

    # Current mission state
    current_mission = Column(String(128), default="EVA-01 · Jezero delta")
    selected_area = Column(JSON, nullable=True)  # {lat, lon, name, radius}
    current_route = Column(JSON, nullable=True)  # Route analysis data
    route_comparisons = Column(JSON, nullable=True)  # List of compared routes
    selected_layers = Column(JSON, nullable=True)  # Active map layers
    scientific_analyses = Column(JSON, nullable=True)  # Previous analyses
    mission_objectives = Column(JSON, nullable=True)  # Current objectives

    # Map state
    map_view = Column(String(16), default="2d")  # 2d or 3d
    map_center = Column(JSON, nullable=True)  # {lat, lon, zoom}
    selected_point = Column(JSON, nullable=True)  # Currently selected location

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="mission_context")


class Document(Base):
    """Ingested documents for RAG."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(256), nullable=False)
    filepath = Column(String(512), nullable=False)
    content_hash = Column(String(64), unique=True, index=True)
    title = Column(String(256), nullable=True)
    source_type = Column(String(64), nullable=True)  # pdf, txt, md, json, url
    metadata = Column("metadata", JSON, nullable=True)
    chunk_count = Column(Integer, default=0)
    is_processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Document chunks for vector search."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_id = Column(String(64), nullable=True, index=True)  # ChromaDB ID
    metadata = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


# Database setup
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()