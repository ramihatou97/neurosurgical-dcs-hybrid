"""
SQLAlchemy database models for persistent storage

Provides ORM models for:
- User authentication and authorization
- Processing sessions
- Document cache metadata
- Uncertainties and resolutions
- Learning patterns
- Audit logging
- Performance metrics

Design: PostgreSQL schema from Phase 1 planning
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey, Numeric, Text, TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid as uuid_lib

# ============================================================================
# PLATFORM-INDEPENDENT UUID TYPE
# ============================================================================

class UUID(TypeDecorator):
    """
    Platform-independent UUID type.

    Uses PostgreSQL's native UUID type in production, falls back to CHAR(32)
    for SQLite in testing. This allows the same models to work in both environments.

    Critical for testing: SQLite doesn't support PostgreSQL's UUID type.
    """
    impl = CHAR
    cache_ok = True

    def __init__(self, as_uuid=True):
        """
        Initialize UUID type

        Args:
            as_uuid: Whether to return UUID objects (ignored, always True for compatibility)
        """
        self.as_uuid = as_uuid
        super().__init__()

    def load_dialect_impl(self, dialect):
        """Load appropriate type for the dialect"""
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(pgUUID(as_uuid=True))
        else:
            # SQLite and other databases use CHAR(32)
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        """Process value being sent to database"""
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            # PostgreSQL handles UUID objects directly
            return str(value) if not isinstance(value, str) else value
        else:
            # SQLite: store as hex string
            if isinstance(value, uuid_lib.UUID):
                return value.hex
            return value

    def process_result_value(self, value, dialect):
        """Process value being retrieved from database"""
        if value is None:
            return value
        if isinstance(value, uuid_lib.UUID):
            return value
        # Convert string back to UUID
        if isinstance(value, str):
            return uuid_lib.UUID(value) if len(value) == 36 else uuid_lib.UUID(hex=value)
        return value


Base = declarative_base()


# ============================================================================
# USER MANAGEMENT
# ============================================================================

class User(Base):
    """
    User model for authentication and authorization

    Stores user credentials, roles, and permissions for RBAC.
    From: complete_1 enhanced API OAuth2 implementation
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    department = Column(String(100))
    role = Column(String(50))  # attending, resident, nurse, admin
    permissions = Column(JSON)  # {"read": true, "write": true, "approve": false}
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    sessions = relationship("ProcessingSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    resolved_uncertainties = relationship("Uncertainty", back_populates="resolver", foreign_keys="Uncertainty.resolved_by")
    learning_patterns = relationship("LearningPattern", back_populates="creator", foreign_keys="LearningPattern.created_by", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


# ============================================================================
# PROCESSING SESSIONS
# ============================================================================

class ProcessingSession(Base):
    """
    Processing session tracking

    Tracks each discharge summary generation session with metadata,
    status, and relationships to documents/uncertainties.
    """
    __tablename__ = 'processing_sessions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    status = Column(String(50))  # 'processing', 'completed', 'error', 'pending_review'
    document_count = Column(Integer)
    confidence_score = Column(Numeric(5, 4))
    requires_review = Column(Boolean, default=False)
    custom_metadata = Column(JSON)  # Additional session-specific metadata (renamed from metadata)
    result_data = Column(JSON)  # Complete processing result (summary, timeline, metrics)

    # Relationships
    user = relationship("User", back_populates="sessions")
    documents = relationship("Document", back_populates="session", cascade="all, delete-orphan")
    uncertainties = relationship("Uncertainty", back_populates="session", cascade="all, delete-orphan")
    metrics = relationship("ProcessingMetric", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ProcessingSession(id={self.id}, user_id={self.user_id}, status='{self.status}')>"


# ============================================================================
# DOCUMENT CACHE
# ============================================================================

class Document(Base):
    """
    Document cache metadata

    Stores document hashes and extraction cache for performance optimization.
    Links to processing sessions for audit trail.
    """
    __tablename__ = 'documents'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('processing_sessions.id'), nullable=False, index=True)
    doc_hash = Column(String(64), unique=True, nullable=False, index=True)  # MD5 hash of content
    doc_type = Column(String(50))  # admission, operative, progress, etc.
    content_summary = Column(Text)  # First 500 chars for reference
    processed_at = Column(DateTime, default=datetime.utcnow, index=True)
    cache_expiry = Column(DateTime)  # TTL for cache
    extraction_cache = Column(JSON)  # Cached extracted facts
    custom_metadata = Column(JSON)  # Additional document metadata (renamed from metadata)

    # Relationships
    session = relationship("ProcessingSession", back_populates="documents")

    def __repr__(self):
        return f"<Document(id={self.id}, doc_hash='{self.doc_hash[:8]}...', doc_type='{self.doc_type}')>"


# ============================================================================
# UNCERTAINTIES
# ============================================================================

class Uncertainty(Base):
    """
    Uncertainty tracking and resolution

    Stores identified uncertainties, their context, and resolution workflow.
    Links to sessions and resolving users for complete audit trail.
    """
    __tablename__ = 'uncertainties'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('processing_sessions.id'), nullable=False, index=True)
    uncertainty_type = Column(String(100))  # CONFLICTING_INFORMATION, MISSING_INFORMATION, etc.
    description = Column(Text)
    conflicting_sources = Column(JSON)  # List of source document identifiers
    suggested_resolution = Column(Text)
    severity = Column(String(20), index=True)  # HIGH, MEDIUM, LOW
    context = Column(JSON)  # Additional context for the uncertainty
    resolved = Column(Boolean, default=False, index=True)
    resolved_by = Column(Integer, ForeignKey('users.id'))
    resolved_at = Column(DateTime)
    resolution = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ProcessingSession", back_populates="uncertainties")
    resolver = relationship("User", back_populates="resolved_uncertainties", foreign_keys=[resolved_by])

    def __repr__(self):
        return f"<Uncertainty(id={self.id}, type='{self.uncertainty_type}', resolved={self.resolved})>"


# ============================================================================
# LEARNING PATTERNS
# ============================================================================

class LearningPattern(Base):
    """
    Learning patterns from uncertainty resolutions

    Stores physician corrections as learning patterns for continuous improvement.
    Tracks success rate and application count for pattern quality assessment.
    """
    __tablename__ = 'learning_patterns'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    pattern_hash = Column(String(64), unique=True, nullable=False, index=True)  # Unique pattern identifier
    fact_type = Column(String(100), index=True)  # medication, temporal_reference, etc.
    original_pattern = Column(Text)  # Original (incorrect) extraction
    correction = Column(Text)  # Corrected version
    context = Column(JSON)  # Context for pattern matching
    success_rate = Column(Numeric(5, 4), default=1.0)  # 0.0-1.0
    applied_count = Column(Integer, default=0)  # How many times applied
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)

    # NEW: Approval workflow
    approved = Column(Boolean, default=False, index=True)  # Requires admin approval before use
    approved_by = Column(Integer, ForeignKey('users.id'))
    approved_at = Column(DateTime)

    # Relationships
    creator = relationship("User", back_populates="learning_patterns", foreign_keys=[created_by])
    approver = relationship("User", foreign_keys=[approved_by])

    def __repr__(self):
        return f"<LearningPattern(id={self.id}, fact_type='{self.fact_type}', approved={self.approved})>"


# ============================================================================
# AUDIT LOG
# ============================================================================

class AuditLog(Base):
    """
    Audit log for compliance and security

    Tracks all user actions with timestamps, IP addresses, and detailed context.
    Essential for HIPAA compliance and security monitoring.
    """
    __tablename__ = 'audit_log'

    id = Column(Integer, primary_key=True)  # Changed from BigInteger for SQLite compatibility
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    action = Column(String(100), index=True)  # PROCESS_DOCUMENTS, RESOLVE_UNCERTAINTY, etc.
    resource_type = Column(String(100))  # session, uncertainty, learning_pattern
    resource_id = Column(UUID(as_uuid=True))
    details = Column(JSON)  # Detailed action context
    ip_address = Column(String(45))  # IP address (IPv4/IPv6) - changed from INET for SQLite compatibility
    user_agent = Column(Text)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(id={self.id}, user_id={self.user_id}, action='{self.action}')>"


# ============================================================================
# PROCESSING METRICS
# ============================================================================

class ProcessingMetric(Base):
    """
    Processing performance metrics

    Stores granular performance metrics for monitoring and optimization.
    Enables analysis of processing times, cache effectiveness, and quality metrics.
    """
    __tablename__ = 'processing_metrics'

    id = Column(Integer, primary_key=True)  # Changed from BigInteger for SQLite compatibility
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('processing_sessions.id'), index=True)
    metric_type = Column(String(50), index=True)  # 'processing_time', 'fact_count', 'cache_hit', etc.
    value = Column(Numeric(10, 4))
    unit = Column(String(20))  # 'ms', 'count', 'percent', etc.
    custom_metadata = Column(JSON)  # Additional metric context (renamed from metadata)

    # Relationships
    session = relationship("ProcessingSession", back_populates="metrics")

    def __repr__(self):
        return f"<ProcessingMetric(id={self.id}, type='{self.metric_type}', value={self.value})>"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_tables(engine):
    """
    Create all tables in the database

    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """
    Drop all tables from the database (use with caution!)

    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.drop_all(engine)


def get_table_names():
    """
    Get list of all table names defined in this module

    Returns:
        List of table name strings
    """
    return [table.name for table in Base.metadata.sorted_tables]
