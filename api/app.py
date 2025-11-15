"""
Enhanced FastAPI Application for Neurosurgical DCS Hybrid System

Provides comprehensive API with:
- OAuth2/JWT authentication (from complete_1) - NOW DB-BACKED
- Processing endpoints (parallel/sequential options)
- Learning system endpoints (submit, approve, review)
- Audit logging for HIPAA compliance - NOW DB-BACKED
- Performance metrics
- WebSocket support for real-time updates

Security: Role-based access control (RBAC)
- read: View summaries
- write: Generate summaries
- approve: Approve learning patterns (admin only)
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, date
from passlib.context import CryptContext
from jose import JWTError, jwt
import logging
import uuid
import os
import json
from dotenv import load_dotenv
from contextlib import contextmanager

# --- DB IMPORTS (Fix #2 + Persistence Fix) ---
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
# --- END DB IMPORTS ---

# Import hybrid engine
import sys
sys.path.insert(0, '..')
from src.engine import HybridNeurosurgicalDCSEngine
# --- DB MODEL IMPORT (Fix #2 + Persistence Fix) ---
from src.database.models import User as UserModel, Base, AuditLog, ProcessingSession as SessionModel
# --- END DB MODEL IMPORT ---


# ========================= CONFIGURATION =========================

load_dotenv()

# --- FIX #2: LOAD FROM ENVIRONMENT ---
SECRET_KEY = os.getenv("SECRET_KEY", "default-fallback-secret-key-CHANGE-ME")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

if "CHANGE-ME" in SECRET_KEY:
    logging.warning("SECURITY WARNING: Using default SECRET_KEY. SET A REAL SECRET_KEY IN YOUR .env FILE.")
# --- END FIX #2 ---

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================= APPLICATION SETUP =========================

app = FastAPI(
    title="Neurosurgical DCS Hybrid API",
    description="Production-grade discharge summary generation with learning system",
    version="3.0.0-hybrid",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize hybrid engine (will be initialized on startup)
engine: Optional[HybridNeurosurgicalDCSEngine] = None

# ========================= DATABASE SESSION (Fix #2) =========================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev_neurosurgical_dcs.db")
db_engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

# Create tables if they don't exist (for all databases)
Base.metadata.create_all(bind=db_engine)

@contextmanager
def get_db_session():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db():
    """Dependency for FastAPI routes to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========================= REMOVED IN-MEMORY STORES =========================
# --- FIX: REMOVED USER_DATABASE DICTIONARY ---
# --- FIX: REMOVED AUDIT_LOG = [] ---
# --- FIX: REMOVED PROCESSING_SESSIONS = {} ---

# ========================= PYDANTIC MODELS =========================

class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: Dict[str, Any]

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    id: int  # ADDED: Required for foreign key relationships
    username: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    permissions: List[str] = []

class ProcessRequest(BaseModel):
    documents: List[Dict]
    options: Dict = Field(default_factory=dict)
    use_parallel: bool = True
    use_cache: bool = True
    apply_learning: bool = True

class LearningFeedbackRequest(BaseModel):
    uncertainty_id: str
    original_extraction: str
    correction: str
    context: Dict
    apply_immediately: bool = False

class LearningPatternApproval(BaseModel):
    pattern_id: str
    approved: bool  # True = approve, False = reject
    reason: Optional[str] = None  # For rejection

class BulkImportRequest(BaseModel):
    bulk_text: str
    separator_type: str = 'auto'  # auto, triple_dash, custom
    custom_separator: Optional[str] = None

class SuggestedDocument(BaseModel):
    content: str
    doc_type: Optional[str] = None
    confidence: float = 0.0
    date: Optional[str] = None
    author: Optional[str] = None

class BulkImportResponse(BaseModel):
    status: str
    suggestedDocuments: List[Dict[str, Any]]
    totalCount: int
    warnings: List[str] = []
    separatorUsed: str
    metadata: Dict[str, Any]

class SessionDetailResponse(BaseModel):
    sessionId: str
    status: str
    createdAt: str
    completedAt: Optional[str] = None
    documentCount: int
    confidenceScore: Optional[float] = None
    requiresReview: bool
    resultData: Dict[str, Any]  # Complete processing result

    class Config:
        from_attributes = True

# ========================= UTILITY FUNCTIONS =========================

def serialize_for_json(obj):
    """
    Recursively convert Python objects to JSON-serializable format

    Handles:
    - datetime.datetime → ISO 8601 string
    - datetime.date → ISO 8601 string
    - dict → recursively process values
    - list → recursively process items
    - other types → pass through
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj

# ========================= AUTHENTICATION FUNCTIONS (Fix #2) =========================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    if isinstance(plain_password, str):
        plain_password = plain_password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str, db: Session) -> Optional[UserModel]:
    """Get user from database"""
    return db.query(UserModel).filter(UserModel.username == username).first()

def authenticate_user(username: str, password: str, db: Session) -> Optional[UserModel]:
    """Authenticate user with username and password from DB"""
    user = get_user(username, db)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Get current user from JWT token and DB"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(username=token_data.username, db=db)
    if user is None:
        raise credentials_exception

    # Convert SQLAlchemy UserModel to Pydantic User model
    return User(
        id=user.id,  # ADDED: Include ID for foreign keys
        username=user.username,
        full_name=user.full_name,
        email=user.email,
        department=user.department,
        role=user.role,
        permissions=user.permissions if user.permissions else []
    )

def check_permission(user: User, permission: str):
    """Check if user has specific permission"""
    if permission not in user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User does not have '{permission}' permission"
        )
    return True

# ========================= AUDIT LOGGING (PERSISTENCE FIX) =========================

def log_audit_event(user: User, action: str, details: Dict, db: Session, request: Optional[Request] = None):
    """Log audit event for compliance to the database"""
    try:
        ip_address = request.client.host if request else None
        user_agent = request.headers.get("user-agent") if request else None

        log_entry = AuditLog(
            user_id=user.id,  # Use the ID from the Pydantic User model
            action=action,
            resource_type=details.get("resource_type"),
            resource_id=details.get("resource_id"),
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        db.add(log_entry)
        db.commit()
        logger.info(f"Audit: {user.username} - {action}")
    except Exception as e:
        logger.error(f"Failed to write audit log to database: {e}")
        db.rollback()

# ========================= STARTUP/SHUTDOWN =========================

@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    global engine

    # Create a default admin user if one doesn't exist (for dev)
    with get_db_session() as db:
        admin = get_user("admin", db)
        if not admin:
            logger.info("Creating default 'admin' user with password 'admin123'")
            admin_user = UserModel(
                username="admin",
                full_name="System Administrator",
                email="admin@hospital.org",
                hashed_password=pwd_context.hash("admin123"),
                department="it",
                role="admin",
                permissions=["read", "write", "approve", "manage"],
                is_active=True
            )
            db.add(admin_user)
            db.commit()

    # Initialize the main engine
    engine = HybridNeurosurgicalDCSEngine(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        enable_learning=True
    )
    await engine.initialize()
    logger.info("✅ Hybrid DCS Engine initialized and ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown engine gracefully"""
    global engine
    if engine:
        await engine.shutdown()
        logger.info("Engine shutdown complete")

# ========================= AUTHENTICATION ENDPOINTS (Fix #2) =========================

@app.post("/api/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login endpoint - generates JWT token

    Returns access token for authenticated requests.
    """
    user = authenticate_user(form_data.username, form_data.password, db)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )

    logger.info(f"User logged in: {user.username}")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": {
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
            "permissions": user.permissions
        }
    }

@app.get("/api/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current authenticated user information"""
    return current_user

# ========================= PROCESSING ENDPOINTS (PERSISTENCE FIX) =========================

@app.post("/api/process")
async def process_documents(
    request_data: ProcessRequest,
    background_tasks: BackgroundTasks,
    request: Request,  # ADDED for IP/User-Agent
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)  # ADDED for DB session
):
    """
    Main processing endpoint - generate discharge summary

    Requires: 'write' permission

    Process:
    1. Validate request
    2. Process documents (parallel/sequential)
    3. Apply learning (if enabled)
    4. Validate results
    5. Log audit event
    6. Return result
    """
    # Check permission
    check_permission(current_user, "write")

    try:
        # Log audit event to database
        log_audit_event(current_user, "PROCESS_DOCUMENTS", {
            "resource_type": "processing_session",
            "document_count": len(request_data.documents),
            "use_parallel": request_data.use_parallel,
            "use_cache": request_data.use_cache,
            "apply_learning": request_data.apply_learning
        }, db, request)

        # Process documents
        result = await engine.process_hospital_course(
            documents=request_data.documents,
            use_parallel=request_data.use_parallel,
            use_cache=request_data.use_cache,
            apply_learning=request_data.apply_learning
        )

        # Create ProcessingSession in database
        session_id = uuid.uuid4()
        db_session = SessionModel(
            id=session_id,
            user_id=current_user.id,
            status='completed' if not result.get('requires_review') else 'pending_review',
            document_count=len(request_data.documents),
            confidence_score=result.get('confidence_score'),
            requires_review=result.get('requires_review', False),
            custom_metadata={
                "use_parallel": request_data.use_parallel,
                "use_cache": request_data.use_cache,
                "apply_learning": request_data.apply_learning
            },
            result_data=serialize_for_json(result)  # Convert datetime objects to ISO strings
        )
        db.add(db_session)
        db.commit()

        result['sessionId'] = str(session_id)  # camelCase to match frontend

        return result

    except Exception as e:
        logger.error(f"Processing error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ========================= BULK IMPORT ENDPOINTS =========================

@app.post("/api/bulk-import/parse", response_model=BulkImportResponse)
async def parse_bulk_documents(
    bulk_request: BulkImportRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Parse bulk text into suggested documents (SAFETY: parse only, no processing)

    Splits bulk text by separator and extracts metadata.
    Returns suggestions that require user verification before processing.

    Requires: 'write' permission
    """
    check_permission(current_user, "write")

    try:
        import re
        from datetime import datetime

        warnings = []
        suggested_documents = []

        # Determine separator
        separator = None
        if bulk_request.separator_type == 'auto':
            # Auto-detect triple dash separator
            if '---' in bulk_request.bulk_text:
                separator = '---'
            elif '\n\n\n' in bulk_request.bulk_text:
                separator = '\n\n\n'
            else:
                warnings.append("No clear separator found. Using double newline as fallback.")
                separator = '\n\n'
        elif bulk_request.separator_type == 'triple_dash':
            separator = '---'
        elif bulk_request.separator_type == 'custom' and bulk_request.custom_separator:
            separator = bulk_request.custom_separator
        else:
            raise HTTPException(status_code=400, detail="Invalid separator configuration")

        # Split text into chunks
        chunks = [chunk.strip() for chunk in bulk_request.bulk_text.split(separator) if chunk.strip()]

        if len(chunks) == 0:
            return BulkImportResponse(
                status="error",
                suggestedDocuments=[],
                totalCount=0,
                warnings=["No documents found in bulk text"],
                separatorUsed=separator,
                metadata={
                    'processorVersion': '1.0',
                    'processingTimeMs': 0,
                    'separatorType': bulk_request.separator_type
                }
            )

        # Document type keywords for detection
        doc_type_patterns = {
            'Admission Note': r'(?i)(admission|admit|admitting)',
            'Progress Note': r'(?i)(progress|daily|soap)',
            'Operative Note': r'(?i)(operative|surgery|procedure|operation)',
            'Consult Note': r'(?i)(consult|consultation)',
            'Discharge Summary': r'(?i)(discharge|summary)',
            'Imaging Report': r'(?i)(ct|mri|x-ray|imaging|radiology)',
            'Lab Report': r'(?i)(lab|laboratory|pathology)',
        }

        # Date patterns (common formats)
        date_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|\w+ \d{1,2},? \d{4})'

        # Process each chunk
        for idx, chunk in enumerate(chunks):
            doc_type = None
            confidence = 0.0
            detected_date = None

            # Detect document type
            for dtype, pattern in doc_type_patterns.items():
                if re.search(pattern, chunk[:500]):  # Check first 500 chars
                    doc_type = dtype
                    confidence = 0.7  # Moderate confidence
                    break

            # Extract date
            date_match = re.search(date_pattern, chunk[:300])
            if date_match:
                detected_date = date_match.group(1)

            # Create suggested document
            suggested_documents.append({
                'index': idx,
                'content': chunk,
                'suggestedType': doc_type,
                'typeConfidence': confidence,
                'detectedDate': detected_date,
                'detectedAuthor': None,  # Could be enhanced with author detection
                'separatorUsed': separator,
                'warnings': []
            })

        logger.info(f"Bulk parse: {len(suggested_documents)} documents suggested by {current_user.username}")

        return BulkImportResponse(
            status="success",
            suggestedDocuments=suggested_documents,
            totalCount=len(suggested_documents),
            warnings=warnings,
            separatorUsed=separator,
            metadata={
                'processorVersion': '1.0',
                'processingTimeMs': 0,  # Could be enhanced with actual timing
                'separatorType': bulk_request.separator_type
            }
        )

    except Exception as e:
        logger.error(f"Bulk import parse error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================= SESSION ENDPOINTS =========================

@app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve a processing session by ID

    Returns complete session data including the generated summary,
    timeline, uncertainties, and metrics.

    Requires: User must own the session
    """
    check_permission(current_user, "read")

    try:
        # Parse UUID
        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        # Query session
        db_session = db.query(SessionModel).filter(
            SessionModel.id == session_uuid,
            SessionModel.user_id == current_user.id  # Security: user can only access their own sessions
        ).first()

        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Return session data
        return SessionDetailResponse(
            sessionId=str(db_session.id),
            status=db_session.status,
            createdAt=db_session.created_at.isoformat() if db_session.created_at else None,
            completedAt=db_session.completed_at.isoformat() if db_session.completed_at else None,
            documentCount=db_session.document_count,
            confidenceScore=float(db_session.confidence_score) if db_session.confidence_score else None,
            requiresReview=db_session.requires_review,
            resultData=db_session.result_data or {}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================= LEARNING SYSTEM ENDPOINTS (Fix #1 + Persistence Fix) =========================

@app.post("/api/learning/feedback")
async def submit_learning_feedback(
    feedback_request: LearningFeedbackRequest,
    request: Request,  # ADDED for IP/User-Agent
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)  # ADDED for DB session
):
    """
    Submit learning feedback from uncertainty resolution

    Creates PENDING learning pattern (requires approval before application)

    Requires: 'write' permission
    """
    check_permission(current_user, "write")

    try:
        # Add feedback (creates PENDING pattern)
        pattern_id = engine.feedback_manager.add_feedback(
            uncertainty_id=feedback_request.uncertainty_id,
            original_extraction=feedback_request.original_extraction,
            correction=feedback_request.correction,
            context=feedback_request.context,
            created_by=current_user.username
        )

        # Log audit event to database
        log_audit_event(current_user, "SUBMIT_LEARNING_FEEDBACK", {
            "resource_type": "learning_pattern",
            "resource_id": pattern_id,
            "pattern_id": pattern_id[:8],
            "uncertainty_id": feedback_request.uncertainty_id,
            "fact_type": feedback_request.context.get('fact_type')
        }, db, request)

        # --- FIX #1: Use feedback_request and check 'approve' permission ---
        if feedback_request.apply_immediately and "approve" in current_user.permissions:
        # --- END FIX #1 ---
            # Admin or user with 'approve' permission can immediately approve
            engine.feedback_manager.approve_pattern(pattern_id, approved_by=current_user.username)
            logger.info(f"User {current_user.username} auto-approved pattern {pattern_id[:8]}")

            # Log the auto-approval
            log_audit_event(current_user, "APPROVE_LEARNING_PATTERN", {
                "resource_type": "learning_pattern",
                "resource_id": pattern_id,
                "pattern_id": pattern_id[:8],
                "auto_approved": True
            }, db, request)

            return {
                "status": "success",
                "pattern_id": pattern_id,
                "pattern_status": "APPROVED",
                "message": "Learning feedback submitted and auto-approved."
            }

        return {
            "status": "success",
            "pattern_id": pattern_id,
            "pattern_status": "PENDING_APPROVAL",
            "message": "Learning feedback submitted. Awaiting admin approval before automatic application."
        }

    except Exception as e:
        logger.error(f"Learning feedback error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning/approve")
async def approve_learning_pattern(
    approval_request: LearningPatternApproval,
    request: Request,  # ADDED for IP/User-Agent
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)  # ADDED for DB session
):
    """
    Approve or reject learning pattern

    **Requires: 'approve' permission (admin only)**

    This is the critical safety gate - only admins can approve patterns
    for automatic application to future extractions.
    """
    # Check admin permission
    check_permission(current_user, "approve")

    try:
        if approval_request.approved:
            # Approve pattern
            success = engine.feedback_manager.approve_pattern(
                pattern_id=approval_request.pattern_id,
                approved_by=current_user.username
            )

            action = "APPROVE_LEARNING_PATTERN"
            message = f"Pattern {approval_request.pattern_id[:8]} approved - will be applied to future extractions"

        else:
            # Reject pattern
            success = engine.feedback_manager.reject_pattern(
                pattern_id=approval_request.pattern_id,
                rejected_by=current_user.username,
                reason=approval_request.reason
            )

            action = "REJECT_LEARNING_PATTERN"
            message = f"Pattern {approval_request.pattern_id[:8]} rejected - will NOT be applied"

        if not success:
            raise HTTPException(status_code=404, detail="Pattern not found")

        # Log audit event to database
        log_audit_event(current_user, action, {
            "resource_type": "learning_pattern",
            "resource_id": approval_request.pattern_id,
            "pattern_id": approval_request.pattern_id[:8],
            "approved": approval_request.approved,
            "reason": approval_request.reason
        }, db, request)

        return {
            "status": "success",
            "action": "approved" if approval_request.approved else "rejected",
            "pattern_id": approval_request.pattern_id,
            "message": message
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern approval error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/pending")
async def get_pending_patterns(current_user: User = Depends(get_current_user)):
    """
    Get pending learning patterns awaiting approval

    Requires: 'approve' permission (admin only)

    Returns list for admin review in learning pattern viewer.
    """
    check_permission(current_user, "approve")

    try:
        pending = engine.feedback_manager.get_pending_patterns()

        return {
            "status": "success",
            "pending_count": len(pending),
            "patterns": pending
        }

    except Exception as e:
        logger.error(f"Get pending patterns error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/approved")
async def get_approved_patterns(current_user: User = Depends(get_current_user)):
    """
    Get approved learning patterns currently in use

    Requires: 'read' permission

    Returns patterns with statistics (application count, success rate).
    """
    check_permission(current_user, "read")

    try:
        approved = engine.feedback_manager.get_approved_patterns()

        return {
            "status": "success",
            "approved_count": len(approved),
            "patterns": approved
        }

    except Exception as e:
        logger.error(f"Get approved patterns error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/statistics")
async def get_learning_statistics(current_user: User = Depends(get_current_user)):
    """
    Get learning system statistics

    Requires: 'read' permission

    Returns comprehensive statistics about learning system.
    """
    check_permission(current_user, "read")

    try:
        stats = engine.feedback_manager.get_statistics()

        return {
            "status": "success",
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Get learning statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================= SYSTEM ENDPOINTS =========================

@app.get("/api/system/statistics")
async def get_system_statistics(current_user: User = Depends(get_current_user)):
    """
    Get system-wide statistics

    Requires: 'read' permission
    """
    check_permission(current_user, "read")

    try:
        stats = engine.get_engine_statistics()

        return {
            "status": "success",
            "statistics": stats,
            "engine_version": engine.get_version(),
            "cache_available": engine.is_cache_available(),
            "learning_enabled": engine.is_learning_enabled()
        }

    except Exception as e:
        logger.error(f"Get system statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/health")
async def health_check():
    """
    Health check endpoint (no authentication required)

    Returns engine status and availability.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": engine.get_version() if engine else "unknown",
        "cache_available": engine.is_cache_available() if engine else False,
        "learning_enabled": engine.is_learning_enabled() if engine else False
    }

# --- FIX: READ AUDIT LOG FROM DATABASE ---
@app.get("/api/audit-log")
async def get_audit_log(
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get audit log entries from the database

    Requires: 'approve' permission (admin only)

    HIPAA compliance: tracks all user actions with full persistence.
    """
    check_permission(current_user, "approve")

    try:
        total_entries = db.query(AuditLog).count()
        recent_entries = db.query(AuditLog).order_by(desc(AuditLog.timestamp)).limit(limit).all()

        return {
            "status": "success",
            "entry_count": len(recent_entries),
            "total_entries": total_entries,
            "entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "user_id": entry.user_id,
                    "action": entry.action,
                    "resource_type": entry.resource_type,
                    "details": entry.details,
                    "ip_address": entry.ip_address,
                    "user_agent": entry.user_agent
                } for entry in recent_entries
            ]
        }
    except Exception as e:
        logger.error(f"Failed to retrieve audit log: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit log")

# ========================= ROOT ENDPOINT =========================

@app.get("/")
async def root():
    """API root - returns welcome message"""
    return {
        "message": "Neurosurgical DCS Hybrid API",
        "version": "3.0.0-hybrid",
        "status": "operational",
        "documentation": "/api/docs",
        "authentication_required": True,
        "features": [
            "Hybrid extraction (complete_1 + v2)",
            "6-stage validation pipeline",
            "NEW contradiction detection",
            "Learning system with approval workflow",
            "Parallel processing",
            "Multi-level caching",
            "Database-backed user authentication",
            "Database-backed audit logging (HIPAA compliant)",
            "Database-backed processing sessions"
        ]
    }

# ========================= ERROR HANDLERS =========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ========================= DEVELOPMENT ENDPOINTS =========================

if __name__ == "__main__":
    import uvicorn
    # This block is for direct execution, not for production
    # Admin user creation moved to startup_event for consistency
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
