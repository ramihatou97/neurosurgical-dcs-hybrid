"""
Unified Hybrid Engine - Main Orchestrator

Orchestrates all components of the hybrid discharge summarizer system:
- Parallel document processing (extraction with caching)
- Learning correction application (approved patterns only)
- Temporal resolution and timeline building
- Comprehensive 6-stage validation
- Performance metrics collection

This is the main entry point for the entire system.

Design: Combines complete_1's robust logic with v2's performance enhancements
"""

import logging
import time
from typing import List, Dict, Optional
from datetime import datetime

from .core.data_models import (
    DischargeSummaryOutput,
    ProcessingMetrics,
    ClinicalTimeline,
    uncertainty_to_dict,
    fact_to_dict
)
from .extraction.fact_extractor import HybridFactExtractor
from .processing.parallel_processor import ParallelProcessor
from .processing.timeline_builder import EnhancedTimelineBuilder
from .processing.validator import ComprehensiveValidator
from .learning.feedback_manager import FeedbackManager
from .cache.redis_manager import RedisCacheManager

logger = logging.getLogger(__name__)


class HybridNeurosurgicalDCSEngine:
    """
    Unified Hybrid Discharge Summary Engine

    Combines:
    - complete_1: Proven narrative generation, uncertainty management, security
    - v2: Parallel processing, temporal reasoning, learning system, caching

    Usage:
        engine = HybridNeurosurgicalDCSEngine(redis_url="redis://localhost:6379")
        result = await engine.process_hospital_course(documents, use_parallel=True)
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        enable_learning: bool = True
    ):
        """
        Initialize hybrid engine with all components

        Args:
            redis_url: Redis connection URL (optional, graceful degradation)
            enable_learning: Whether to enable learning system (default: True)
        """
        logger.info("=" * 70)
        logger.info("Initializing Hybrid Neurosurgical DCS Engine")
        logger.info("=" * 70)

        # Initialize cache manager (optional, graceful degradation)
        self.cache_manager: Optional[RedisCacheManager] = None
        if redis_url:
            self.cache_manager = RedisCacheManager(redis_url)
            logger.info(f"Redis cache manager created: {redis_url}")
        else:
            logger.info("Running without Redis cache (graceful degradation)")

        # Initialize core components
        self.extractor = HybridFactExtractor()
        self.parallel_processor = ParallelProcessor(cache_manager=self.cache_manager)
        self.timeline_builder = EnhancedTimelineBuilder()
        self.validator = ComprehensiveValidator()

        # Initialize learning system
        self.enable_learning = enable_learning
        if enable_learning:
            self.feedback_manager = FeedbackManager()
            logger.info("Learning system enabled (approval workflow active)")
        else:
            self.feedback_manager = None
            logger.info("Learning system disabled")

        # Performance tracking
        self.total_processed = 0
        self.total_processing_time_ms = 0

        logger.info("✅ Hybrid engine initialized successfully")
        logger.info("=" * 70)

    # ========================================================================
    # MAIN PROCESSING ENTRY POINTS
    # ========================================================================

    async def process_hospital_course(
        self,
        documents: List[Dict],
        use_parallel: bool = True,
        use_cache: bool = True,
        apply_learning: bool = True
    ) -> Dict:
        """
        Main entry point: Process hospital course documents

        Complete Pipeline:
        1. Parallel document extraction (with caching)
        2. Apply approved learning corrections
        3. Temporal resolution and timeline building
        4. Comprehensive 6-stage validation
        5. Generate output with metrics

        Args:
            documents: List of document dictionaries
            use_parallel: Use parallel processing (default: True)
            use_cache: Use Redis caching if available (default: True)
            apply_learning: Apply approved learning patterns (default: True)

        Returns:
            Dictionary with discharge summary, timeline, uncertainties, metrics

        Processing Time Target: <1s with cache, <8s without (parallel)
        """
        logger.info(f"Processing {len(documents)} documents "
                   f"(parallel={use_parallel}, cache={use_cache}, learning={apply_learning})")

        start_time = time.time()

        # ====================================================================
        # Connect to Redis if needed
        # ====================================================================
        if use_cache and self.cache_manager and not self.cache_manager.is_connected():
            await self.cache_manager.connect()

        # ====================================================================
        # Check complete result cache first
        # ====================================================================
        if use_cache and self.cache_manager and self.cache_manager.is_connected():
            result_hash = RedisCacheManager.generate_result_hash(documents)
            cached_result = await self.cache_manager.get_complete_result(result_hash)

            if cached_result:
                logger.info(f"🚀 Complete result loaded from cache! (major speedup)")
                cached_result['from_cache'] = True
                return cached_result

        # ====================================================================
        # STEP 1: Parallel Document Extraction
        # ====================================================================
        logger.info("Step 1: Parallel document extraction")
        step_start = time.time()

        facts, classified_docs, metrics = await self.parallel_processor.process_documents_parallel(
            documents,
            use_cache=use_cache
        )

        logger.info(f"Extracted {len(facts)} facts from {len(classified_docs)} documents "
                   f"in {(time.time() - step_start)*1000:.1f}ms")

        # ====================================================================
        # STEP 2: Apply Approved Learning Corrections (if enabled)
        # ====================================================================
        if apply_learning and self.enable_learning and self.feedback_manager:
            logger.info("Step 2: Applying approved learning corrections")
            step_start = time.time()

            # Load patterns from cache if available
            if use_cache and self.cache_manager:
                await self.feedback_manager.load_from_redis(self.cache_manager)

            # Apply corrections
            facts = self.feedback_manager.apply_corrections(facts)

            corrections_applied = sum(1 for f in facts if f.correction_applied)
            logger.info(f"Applied {corrections_applied} approved learning corrections "
                       f"in {(time.time() - step_start)*1000:.1f}ms")

            metrics.learning_patterns_applied = corrections_applied

        # ====================================================================
        # STEP 3: Sequential Pipeline (timeline → validation)
        # ====================================================================
        logger.info("Step 3: Sequential pipeline (timeline + validation)")
        step_start = time.time()

        timeline, uncertainties, metrics = self.parallel_processor.process_pipeline_sequential(
            facts,
            classified_docs,
            metrics
        )

        logger.info(f"Timeline built and validated in {(time.time() - step_start)*1000:.1f}ms")

        # ====================================================================
        # STEP 4: Prepare Final Output
        # ====================================================================
        total_time = time.time() - start_time
        metrics.total_processing_time_ms = int(total_time * 1000)

        logger.info(f"✅ Processing complete: {total_time*1000:.1f}ms total, "
                   f"{len(uncertainties)} uncertainties, "
                   f"{metrics.uncertainties_detected} validation issues")

        # Build output
        output = self._prepare_output(
            timeline=timeline,
            facts=facts,
            uncertainties=uncertainties,
            metrics=metrics
        )

        # ====================================================================
        # STEP 5: Cache Complete Result
        # ====================================================================
        if use_cache and self.cache_manager and self.cache_manager.is_connected():
            result_hash = RedisCacheManager.generate_result_hash(documents)
            await self.cache_manager.set_complete_result(result_hash, output)
            logger.debug(f"Cached complete result: {result_hash[:8]}")

        # Update engine statistics
        self.total_processed += 1
        self.total_processing_time_ms += metrics.total_processing_time_ms

        return output

    # ========================================================================
    # OUTPUT PREPARATION
    # ========================================================================

    def _prepare_output(
        self,
        timeline: ClinicalTimeline,
        facts: List,
        uncertainties: List,
        metrics: ProcessingMetrics
    ) -> Dict:
        """
        Prepare final output structure

        Args:
            timeline: Clinical timeline
            facts: Extracted facts
            uncertainties: Validation uncertainties
            metrics: Processing metrics

        Returns:
            Complete output dictionary
        """
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(facts, uncertainties)

        # Determine if review required
        requires_review = any(u.severity == "HIGH" for u in uncertainties)

        # Format uncertainties for output
        formatted_uncertainties = [uncertainty_to_dict(u) for u in uncertainties]

        # Build source attribution map
        source_attribution = self._build_source_attribution(facts)

        # Timeline summary
        timeline_summary = self.timeline_builder.get_timeline_summary(timeline)

        # Validation summary
        validation_summary = self.validator.get_validation_summary(uncertainties)

        # Learning statistics (if enabled)
        learning_stats = None
        if self.feedback_manager:
            learning_stats = self.feedback_manager.get_statistics()

        output = {
            # Summary information
            'summary_text': self._generate_summary_text(timeline, facts),

            # Quality metrics
            'confidence_score': confidence_score,
            'requires_review': requires_review,

            # Uncertainties for physician review
            'uncertainties': formatted_uncertainties,
            'uncertainty_count': len(uncertainties),
            'high_severity_count': sum(1 for u in uncertainties if u.severity == "HIGH"),

            # Timeline data
            'timeline_summary': timeline_summary,
            'key_events': timeline.key_events,
            'clinical_progression': timeline.progression,

            # Source attribution
            'source_attribution': source_attribution,

            # Validation report
            'validation_summary': validation_summary,

            # Performance metrics
            'metrics': {
                'total_processing_time_ms': metrics.total_processing_time_ms,
                'extraction_time_ms': metrics.extraction_time_ms,
                'validation_time_ms': metrics.validation_time_ms,
                'documents_processed': metrics.documents_processed,
                'facts_extracted': metrics.facts_extracted,
                'temporal_references_resolved': metrics.temporal_references_resolved,
                'temporal_resolution_accuracy': metrics.temporal_resolution_accuracy,
                'uncertainties_detected': metrics.uncertainties_detected,
                'learning_patterns_applied': metrics.learning_patterns_applied,
                'cache_hit_rate': metrics.cache_hit_rate
            },

            # Learning statistics (if enabled)
            'learning_statistics': learning_stats,

            # Processing metadata
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'engine_version': '3.0.0-hybrid',
                'parallel_processing': True,
                'learning_enabled': self.enable_learning
            },

            # Full timeline (for detailed view)
            'timeline': {
                str(date_key): [fact_to_dict(f) for f in date_facts]
                for date_key, date_facts in timeline.timeline.items()
            }
        }

        return output

    def _generate_summary_text(
        self,
        timeline: ClinicalTimeline,
        facts: List
    ) -> str:
        """
        Generate brief summary text

        Note: Full narrative generation will be implemented in future enhancement
        For now, returns structured summary of key information

        Args:
            timeline: Clinical timeline
            facts: Extracted facts

        Returns:
            Summary text string
        """
        lines = []
        lines.append("DISCHARGE SUMMARY - NEUROSURGICAL SERVICE")
        lines.append("=" * 60)
        lines.append("")

        # Admission information
        if timeline.admission_date:
            lines.append(f"Admission Date: {timeline.admission_date.strftime('%B %d, %Y')}")

        if timeline.discharge_date:
            lines.append(f"Discharge Date: {timeline.discharge_date.strftime('%B %d, %Y')}")

        if timeline.total_hospital_days:
            lines.append(f"Length of Stay: {timeline.total_hospital_days} days")

        lines.append("")

        # Key facts summary
        lines.append("KEY CLINICAL INFORMATION:")
        lines.append("")

        # NOTE: 'diagnosis' fact_type removed - it was never produced by extractors
        # (orphan consumer causing silent empty section - same pattern as previous bug)

        # Procedures
        procedures = [f for f in facts if f.fact_type == 'procedure']
        if procedures:
            lines.append("")
            lines.append("Procedures:")
            for proc in procedures[:3]:  # Top 3 procedures
                lines.append(f"- {proc.fact}")

        # Medications
        meds = [f for f in facts if f.fact_type == 'medication']
        if meds:
            lines.append("")
            lines.append("Discharge Medications:")
            for med in meds[:10]:  # Top 10 medications
                lines.append(f"- {med.fact}")

        # Clinical Findings (from operative notes)
        findings = [f for f in facts if f.fact_type == 'finding']
        if findings:
            lines.append("")
            lines.append("Clinical Findings:")
            for finding in findings[:3]:
                lines.append(f"- {finding.fact}")

        # Clinical Scores
        scores = [f for f in facts if f.fact_type == 'clinical_score']
        if scores:
            lines.append("")
            lines.append("Clinical Assessment:")
            for score in scores:
                lines.append(f"- {score.fact}")

        # Vital Signs (most recent)
        vital_signs = [f for f in facts if f.fact_type == 'vital_sign']
        if vital_signs:
            # Sort by timestamp descending to get most recent
            vital_signs_sorted = sorted(vital_signs, key=lambda f: f.absolute_timestamp, reverse=True)
            lines.append("")
            lines.append("Vital Signs (Most Recent):")
            # Show most recent vitals (up to 5)
            for vs in vital_signs_sorted[:5]:
                lines.append(f"- {vs.fact}")

        # Laboratory Values (show abnormal/critical only)
        labs = [f for f in facts if f.fact_type == 'lab_value']
        abnormal_labs = [l for l in labs if hasattr(l, 'severity') and l.severity in ['HIGH', 'CRITICAL', 'LOW']]
        if abnormal_labs:
            lines.append("")
            lines.append("Notable Laboratory Values:")
            for lab in abnormal_labs[:5]:
                lines.append(f"- {lab.fact}")

        # Complications
        complications = [f for f in facts if f.fact_type == 'complication']
        if complications:
            lines.append("")
            lines.append("Complications:")
            for comp in complications:
                lines.append(f"- {comp.fact}")

        # Recommendations (from consult notes)
        recommendations = [f for f in facts if f.fact_type == 'recommendation']
        if recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in recommendations[:5]:
                lines.append(f"- {rec.fact}")

        lines.append("")
        lines.append("[Note: Full narrative generation pending - Phase 4 future enhancement]")

        return "\n".join(lines)

    def _calculate_confidence_score(
        self,
        facts: List,
        uncertainties: List
    ) -> float:
        """
        Calculate overall confidence score

        Factors:
        - Average fact confidence
        - Number/severity of uncertainties
        - Completeness

        Args:
            facts: Extracted facts
            uncertainties: Validation uncertainties

        Returns:
            Confidence score (0.0-1.0)
        """
        if not facts:
            return 0.0

        # Base score: average fact confidence
        avg_confidence = sum(f.confidence for f in facts) / len(facts)

        # Penalty for uncertainties
        high_severity_count = sum(1 for u in uncertainties if u.severity == "HIGH")
        medium_severity_count = sum(1 for u in uncertainties if u.severity == "MEDIUM")

        # Penalty: -0.05 per HIGH, -0.02 per MEDIUM
        penalty = (high_severity_count * 0.05) + (medium_severity_count * 0.02)

        final_score = max(0.0, avg_confidence - penalty)

        return round(final_score, 4)

    def _build_source_attribution(self, facts: List) -> Dict:
        """
        Build source attribution map

        Every fact traceable to source document and line number
        for complete transparency and verification.

        Args:
            facts: Extracted facts

        Returns:
            Source attribution dictionary
        """
        attribution = {}

        for fact in facts:
            if fact.fact_id not in attribution:
                attribution[fact.fact_id] = {
                    'fact': fact.fact,
                    'source_document': fact.source_doc,
                    'source_line': fact.source_line,
                    'confidence': fact.confidence,
                    'fact_type': fact.fact_type,
                    'requires_validation': fact.requires_validation
                }

        return attribution

    # ========================================================================
    # ENGINE STATISTICS
    # ========================================================================

    def get_engine_statistics(self) -> Dict:
        """
        Get engine-level statistics

        Returns:
            Dictionary with processing statistics
        """
        avg_processing_time = (
            self.total_processing_time_ms / self.total_processed
            if self.total_processed > 0 else 0
        )

        stats = {
            'total_processed': self.total_processed,
            'total_processing_time_ms': self.total_processing_time_ms,
            'average_processing_time_ms': avg_processing_time,
            'learning_enabled': self.enable_learning,
            'cache_enabled': self.cache_manager is not None
        }

        # Add learning statistics if enabled
        if self.feedback_manager:
            stats['learning_statistics'] = self.feedback_manager.get_statistics()

        # Add cache statistics if available
        if self.cache_manager:
            stats['cache_statistics'] = self.cache_manager.get_cache_stats()

        return stats

    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================

    async def initialize(self):
        """
        Initialize engine (connect to Redis, load learning patterns)

        Call this before processing documents.
        """
        # Connect to Redis
        if self.cache_manager:
            await self.cache_manager.connect()

        # Load learning patterns
        if self.feedback_manager and self.cache_manager:
            await self.feedback_manager.load_from_redis(self.cache_manager)

        logger.info("Engine initialized and ready")

    async def shutdown(self):
        """
        Shutdown engine gracefully (save patterns, close connections)

        Call this when done processing.
        """
        # Save learning patterns
        if self.feedback_manager and self.cache_manager:
            await self.feedback_manager.save_to_redis(self.cache_manager)

        # Close Redis connection
        if self.cache_manager:
            await self.cache_manager.close()

        logger.info("Engine shutdown complete")

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    def get_version(self) -> str:
        """Get engine version"""
        return "3.0.0-hybrid"

    def is_cache_available(self) -> bool:
        """Check if cache is available"""
        return self.cache_manager is not None and self.cache_manager.is_connected()

    def is_learning_enabled(self) -> bool:
        """Check if learning system is enabled"""
        return self.enable_learning and self.feedback_manager is not None
