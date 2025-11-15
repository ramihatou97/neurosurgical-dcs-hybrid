"""
Unified data models combining best aspects from both versions

This module provides the core data structures for the hybrid discharge summarizer system,
integrating ExtractedFact from complete_1 and EnhancedFact from v2 into a unified model.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
import uuid


# ============================================================================
# ENUMERATIONS
# ============================================================================

class DocumentType(Enum):
    """
    Clinical document types
    From: complete_1 engine lines 19-27
    """
    ADMISSION_NOTE = "admission"
    PROGRESS_NOTE = "progress"
    CONSULT_NOTE = "consult"
    OPERATIVE_NOTE = "operative"
    CLINIC_NOTE = "clinic"
    IMAGING_REPORT = "imaging"
    LAB_REPORT = "lab"
    NURSING_NOTE = "nursing"
    DISCHARGE_PLANNING = "discharge_planning"


class FactType(Enum):
    """Types of clinical facts that can be extracted"""
    MEDICATION = "medication"
    LAB_VALUE = "lab_value"
    CLINICAL_SCORE = "clinical_score"
    PROCEDURE = "procedure"
    DIAGNOSIS = "diagnosis"
    COMPLICATION = "complication"
    FINDING = "finding"
    RECOMMENDATION = "recommendation"
    TEMPORAL_REFERENCE = "temporal_reference"
    VITAL_SIGN = "vital_sign"


class SeverityLevel(Enum):
    """Severity classification for clinical findings"""
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class UncertaintyType(Enum):
    """Types of uncertainties that can be detected"""
    CONFLICTING_INFORMATION = "CONFLICTING_INFORMATION"
    MISSING_INFORMATION = "MISSING_INFORMATION"
    TEMPORAL_INCONSISTENCY = "TEMPORAL_INCONSISTENCY"
    DATA_INTEGRITY_ERROR = "DATA_INTEGRITY_ERROR"
    CONTRADICTORY_STATEMENTS = "CONTRADICTORY_STATEMENTS"
    CONTRADICTORY_OUTCOMES = "CONTRADICTORY_OUTCOMES"
    CRITICAL_LAB_VALUE = "CRITICAL_LAB_VALUE"
    INVALID_SCORE_RANGE = "INVALID_SCORE_RANGE"
    EXCESSIVE_MEDICATION_DOSE = "EXCESSIVE_MEDICATION_DOSE"
    DISCHARGE_STATUS_CONTRADICTION = "DISCHARGE_STATUS_CONTRADICTION"


# ============================================================================
# CORE DATA MODELS
# ============================================================================

@dataclass
class ClinicalDocument:
    """
    Represents a clinical document with metadata
    From: complete_1 engine lines 29-38
    """
    doc_type: DocumentType
    timestamp: datetime
    author: str
    specialty: str
    content: str
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate document fields"""
        if not self.content or len(self.content) < 10:
            self.confidence *= 0.5  # Reduce confidence for very short documents
        if "[**redacted**]" in self.content.lower():
            self.confidence *= 0.7  # Reduce confidence for redacted content


@dataclass
class HybridClinicalFact:
    """
    Unified clinical fact model combining ExtractedFact (complete_1) + EnhancedFact (v2)

    Combines:
    - complete_1 ExtractedFact: Core attributes (lines 42-51)
    - v2 EnhancedFact: Enhanced clinical context (lines 39-92)

    This is the fundamental unit of clinical information in the hybrid system.
    """
    # Core attributes (from complete_1)
    fact: str  # The extracted clinical fact text
    source_doc: str  # Document identifier for provenance
    source_line: int  # Line number in source document
    timestamp: datetime  # Document timestamp
    confidence: float  # Extraction confidence (0.0-1.0)
    fact_type: str  # Type of fact (medication, lab, etc.)
    requires_validation: bool = False  # Flag for physician review

    # Enhanced attributes (from v2)
    absolute_timestamp: Optional[datetime] = None  # Resolved temporal reference
    clinical_context: Dict = field(default_factory=dict)  # Additional clinical metadata
    normalized_value: Optional[Any] = None  # Normalized/structured value
    severity: Optional[str] = None  # Severity classification
    clinical_significance: Optional[str] = None  # Clinical importance level
    related_facts: List[str] = field(default_factory=list)  # Related fact IDs

    # Learning integration (new)
    correction_applied: bool = False  # Has a learned correction been applied?
    correction_source: Optional[str] = None  # Pattern ID of applied correction

    # Unique identifier
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Post-initialization validation and defaults"""
        if self.absolute_timestamp is None:
            self.absolute_timestamp = self.timestamp

        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class ClinicalConcept:
    """
    Normalized clinical concept with semantic information
    From: v2 engine lines (used for lab values, medications)

    Provides structured representation of clinical entities with
    normal ranges, classifications, and clinical implications.
    """
    concept_type: str  # lab, medication, procedure, diagnosis
    name: str  # Concept name (e.g., "Sodium", "Nimodipine")
    value: Any  # Actual value
    unit: Optional[str] = None  # Unit of measurement (mg, mmol/L, etc.)
    normal_range: Optional[Tuple[float, float]] = None  # (min, max) for labs
    classification: Optional[str] = None  # Drug class, procedure category, etc.
    severity: Optional[str] = None  # NORMAL, LOW, HIGH, CRITICAL
    clinical_implications: List[str] = field(default_factory=list)  # Clinical significance

    def is_critical(self) -> bool:
        """Check if this concept requires immediate attention"""
        return self.severity == "CRITICAL"

    def is_abnormal(self) -> bool:
        """Check if value is outside normal range"""
        if self.normal_range and isinstance(self.value, (int, float)):
            return not (self.normal_range[0] <= self.value <= self.normal_range[1])
        return False


@dataclass
class ClinicalUncertainty:
    """
    Represents an uncertainty or issue requiring resolution
    From: complete_1 engine lines (enhanced with learning integration)

    Tracks conflicting information, missing data, and other issues
    that require physician review before finalizing the summary.
    """
    issue_type: str  # Type of uncertainty (see UncertaintyType enum)
    description: str  # Human-readable description
    conflicting_sources: List[str]  # Source documents involved
    suggested_resolution: str  # Suggested action to resolve
    severity: str  # HIGH, MEDIUM, LOW
    context: Dict  # Additional context information

    # Unique identifier
    uncertainty_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Resolution tracking
    resolved: bool = False
    resolved_by: Optional[str] = None  # Username of resolver
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None  # Resolution text

    # Learning integration (new)
    resolution_id: Optional[str] = None  # ID linking to resolution
    auto_resolvable: bool = False  # Can be auto-resolved via learning?
    learned_pattern: Optional[str] = None  # Pattern ID if learned

    def mark_resolved(self, resolved_by: str, resolution: str):
        """Mark this uncertainty as resolved"""
        self.resolved = True
        self.resolved_by = resolved_by
        self.resolved_at = datetime.now()
        self.resolution = resolution


@dataclass
class LearningFeedback:
    """
    Learning pattern from uncertainty resolution
    From: v2 engine lines 629-687

    Captures physician corrections to enable continuous learning
    and automatic improvement of future extractions.
    """
    uncertainty_id: str  # ID of original uncertainty
    original_extraction: str  # Original (incorrect) extraction
    correction: str  # Corrected version
    context: Dict  # Context for pattern matching
    timestamp: datetime

    # Performance tracking
    applied_count: int = 0  # How many times applied
    success_rate: float = 1.0  # Success rate (0.0-1.0)

    # Pattern identification
    pattern_hash: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_by: Optional[str] = None  # Username

    def update_success_rate(self, success: bool):
        """Update success rate with exponential moving average"""
        alpha = 0.1  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)


@dataclass
class ClinicalTimeline:
    """
    Represents the temporal structure of a patient's hospital course

    Organizes facts chronologically with clinical progression analysis,
    key events, and anchor events for temporal reasoning.
    """
    # Date-keyed timeline
    timeline: Dict[datetime, List[HybridClinicalFact]] = field(default_factory=dict)

    # Clinical progression tracking
    progression: Dict[str, List[Dict]] = field(default_factory=dict)

    # Key clinical events
    key_events: List[Dict] = field(default_factory=list)

    # Anchor events for temporal reasoning (admission, surgery dates)
    anchor_events: List[Dict] = field(default_factory=list)

    # Timeline metadata
    admission_date: Optional[datetime] = None
    discharge_date: Optional[datetime] = None
    total_hospital_days: int = 0

    def add_fact(self, date: datetime, fact: HybridClinicalFact):
        """Add a fact to the timeline"""
        if date not in self.timeline:
            self.timeline[date] = []
        self.timeline[date].append(fact)

    def get_facts_by_type(self, fact_type: str) -> List[HybridClinicalFact]:
        """Get all facts of a specific type"""
        facts = []
        for date_facts in self.timeline.values():
            facts.extend([f for f in date_facts if f.fact_type == fact_type])
        return facts

    def get_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range of the timeline"""
        if not self.timeline:
            return None, None
        dates = list(self.timeline.keys())
        return min(dates), max(dates)


@dataclass
class ProcessingMetrics:
    """
    Performance and quality metrics for a processing session
    """
    # Processing performance
    total_processing_time_ms: int = 0
    extraction_time_ms: int = 0
    validation_time_ms: int = 0
    narrative_generation_time_ms: int = 0

    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Extraction statistics
    documents_processed: int = 0
    facts_extracted: int = 0
    facts_per_document: float = 0.0

    # Temporal reasoning
    temporal_references_resolved: int = 0
    temporal_resolution_accuracy: float = 0.0

    # Clinical interpretation
    lab_values_normalized: int = 0
    medications_classified: int = 0
    critical_values_detected: int = 0

    # Learning system
    learning_patterns_applied: int = 0
    corrections_applied: int = 0

    # Validation
    uncertainties_detected: int = 0
    high_severity_uncertainties: int = 0

    # Quality scores
    confidence_score: float = 0.0
    completeness_score: float = 0.0

    # Parallel processing
    parallel_speedup: float = 1.0
    tasks_executed_in_parallel: int = 0

    def calculate_cache_hit_rate(self):
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        self.cache_hit_rate = self.cache_hits / total if total > 0 else 0.0

    def calculate_facts_per_document(self):
        """Calculate average facts per document"""
        self.facts_per_document = (
            self.facts_extracted / self.documents_processed
            if self.documents_processed > 0 else 0.0
        )


@dataclass
class DischargeSummaryOutput:
    """
    Final output structure for discharge summary
    """
    # Main summary
    discharge_summary: str

    # Quality metrics
    confidence_score: float
    requires_review: bool

    # Uncertainties
    uncertainties: List[Dict]

    # Source attribution
    source_attribution: Dict

    # Validation report
    validation_report: Dict

    # Processing metadata
    processing_metadata: Dict

    # Session information
    session_id: str

    # Timeline (optional, for detailed view)
    timeline: Optional[ClinicalTimeline] = None

    # Performance metrics
    metrics: Optional[ProcessingMetrics] = None


# ============================================================================
# VALIDATION MODELS
# ============================================================================

@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    score: float  # 0.0-1.0
    issues: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """Specific validation issue"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    location: str  # Where in the document/timeline
    fact_id: Optional[str] = None
    suggested_fix: Optional[str] = None


# ============================================================================
# API MODELS (for request/response)
# ============================================================================

@dataclass
class ProcessRequest:
    """Request model for document processing"""
    documents: List[Dict]
    options: Dict = field(default_factory=dict)
    use_cache: bool = True
    use_parallel: bool = True
    apply_learning: bool = True


@dataclass
class ResolutionRequest:
    """Request model for uncertainty resolution"""
    uncertainty_id: str
    resolution: str
    resolved_by: str
    apply_learning: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _parse_iso_date(date_str: str) -> datetime:
    """
    Parse ISO 8601 date string, handling JavaScript toISOString() format

    JavaScript's toISOString() produces: "2024-01-15T05:00:00.000Z"
    Python's fromisoformat() (3.9) doesn't accept 'Z' suffix

    This function handles both formats:
    - With 'Z': "2024-01-15T05:00:00.000Z" → converts to "+00:00"
    - Without 'Z': "2024-01-15T05:00:00" → passes through

    Args:
        date_str: ISO 8601 formatted date string

    Returns:
        datetime object
    """
    if date_str.endswith('Z'):
        # Replace 'Z' (Zulu time = UTC) with explicit timezone
        date_str = date_str[:-1] + '+00:00'
    return datetime.fromisoformat(date_str)


def create_clinical_document_from_dict(doc_dict: Dict) -> ClinicalDocument:
    """
    Convert dictionary to ClinicalDocument
    Handles various input formats from both systems
    """
    return ClinicalDocument(
        doc_type=DocumentType(doc_dict.get('type', 'progress')),
        timestamp=_parse_iso_date(doc_dict['date']) if 'date' in doc_dict and doc_dict['date'] else datetime.now(),
        author=doc_dict.get('author', 'Unknown'),
        specialty=doc_dict.get('specialty', 'General'),
        content=doc_dict['content'],
        confidence=doc_dict.get('confidence', 1.0),
        metadata=doc_dict.get('metadata', {})
    )


def fact_to_dict(fact: HybridClinicalFact) -> Dict:
    """Convert HybridClinicalFact to dictionary for serialization"""
    return {
        'fact_id': fact.fact_id,
        'fact': fact.fact,
        'source_doc': fact.source_doc,
        'source_line': fact.source_line,
        'timestamp': fact.timestamp.isoformat(),
        'absolute_timestamp': fact.absolute_timestamp.isoformat() if fact.absolute_timestamp else None,
        'confidence': fact.confidence,
        'fact_type': fact.fact_type,
        'requires_validation': fact.requires_validation,
        'clinical_context': fact.clinical_context,
        'normalized_value': str(fact.normalized_value) if fact.normalized_value else None,
        'severity': fact.severity,
        'clinical_significance': fact.clinical_significance,
        'related_facts': fact.related_facts,
        'correction_applied': fact.correction_applied,
        'correction_source': fact.correction_source
    }


def uncertainty_to_dict(uncertainty: ClinicalUncertainty) -> Dict:
    """Convert ClinicalUncertainty to dictionary for API response"""
    return {
        'id': uncertainty.uncertainty_id,
        'type': uncertainty.issue_type,
        'description': uncertainty.description,
        'sources': uncertainty.conflicting_sources,
        'suggestion': uncertainty.suggested_resolution,
        'severity': uncertainty.severity,
        'context': uncertainty.context,
        'resolved': uncertainty.resolved,
        'resolved_by': uncertainty.resolved_by,
        'resolved_at': uncertainty.resolved_at.isoformat() if uncertainty.resolved_at else None,
        'resolution': uncertainty.resolution,
        'ui_display': {
            'color': 'red' if uncertainty.severity == 'HIGH' else 'orange' if uncertainty.severity == 'MEDIUM' else 'yellow',
            'icon': '⚠️' if uncertainty.severity == 'HIGH' else '⚡' if uncertainty.severity == 'MEDIUM' else 'ℹ️',
            'position': 'inline'
        }
    }
