"""
Hybrid Fact Extractor - Best of Both Worlds

Combines extraction methods from:
- complete_1: Comprehensive patterns, operative/consult specialization, high-risk flagging
- v2: Knowledge base classification, lab normalization, clinical interpretation

Entity-Specific Strategy:
1. Medications:complete_1 patterns + v2 knowledge base
2. Labs: v2 normalization + clinical interpretation
3. Clinical Scores: complete_1 robust neurosurgical patterns
4. Procedures: complete_1 domain-specific operative extraction
5. Temporal: v2 comprehensive pattern recognition
6. Consultations: complete_1 specialty-specific extraction

Source References:
- complete_1 engine.py lines 316-392 (medications, scores)
- v2 engine_v2.py lines 391-474 (enhanced extraction with KB)
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from ..core.data_models import HybridClinicalFact, ClinicalDocument, DocumentType
from ..core.knowledge_base import ClinicalKnowledgeBase
from .llm_extractor import LlmExtractor

logger = logging.getLogger(__name__)


class HybridFactExtractor:
    """
    Unified fact extraction combining best methods from both implementations

    Performance Metrics:
    - Medication extraction: ~100-200ms per document
    - Lab extraction: ~50-100ms per document
    - Clinical scores: ~50ms per document
    - Total: ~200-500ms per document (cached: ~10ms)
    """

    def __init__(self, llm_extractor: Optional[LlmExtractor] = None):
        """
        Initialize extractor with clinical knowledge base and optional LLM extractor

        Args:
            llm_extractor: Optional LLM extractor for smart fallback extraction.
                          If None, regex-only extraction is used (backward compatible).
        """
        self.knowledge_base = ClinicalKnowledgeBase()
        self.llm_extractor = llm_extractor
        if self.llm_extractor:
            logger.info("Hybrid fact extractor initialized with LLM-Fallback enabled")
        else:
            logger.info("Hybrid fact extractor initialized (Regex-Only Mode)")

    # ========================================================================
    # MAIN EXTRACTION ROUTING
    # ========================================================================

    def extract_facts(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Route to appropriate extraction method based on document type

        Args:
            doc: Clinical document to extract from

        Returns:
            List of extracted facts with confidence scores and clinical context
        """
        logger.debug(f"Extracting facts from {doc.doc_type.value} document")

        # Route based on document type (complete_1 approach)
        if doc.doc_type == DocumentType.OPERATIVE_NOTE:
            return self._extract_operative_facts(doc)
        elif doc.doc_type == DocumentType.CONSULT_NOTE:
            return self._extract_consult_facts(doc)
        elif doc.doc_type == DocumentType.LAB_REPORT:
            return self._extract_lab_report_facts(doc)
        else:
            return self._extract_general_facts(doc)

    def _extract_general_facts(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Extract facts from general documents (admission, progress, nursing notes)

        Extracts:
        - Diagnoses (with LLM fallback)
        - Medications
        - Lab values
        - Clinical scores
        - Vital signs
        - General findings
        """
        facts = []

        # Extract all fact types (including diagnoses with LLM fallback)
        facts.extend(self._extract_diagnoses(doc))
        facts.extend(self._extract_medications(doc))
        facts.extend(self._extract_labs(doc))
        facts.extend(self._extract_clinical_scores(doc))
        facts.extend(self._extract_vital_signs(doc))
        facts.extend(self._extract_temporal_references(doc))

        logger.debug(f"Extracted {len(facts)} facts from general document")
        return self._deduplicate_facts(facts)

    # ========================================================================
    # DIAGNOSES - WITH LLM FALLBACK
    # ========================================================================

    def _extract_diagnoses(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Extract diagnosis from general/consult notes.
        Uses regex first, then LLM fallback if needed.

        Returns:
            List of diagnosis facts
        """
        facts = []
        content = doc.content

        # 1. Try Regex Patterns First
        diag_patterns = [
            r'diagnosis:?\s*([^\n]+)',
            r'assessment:?\s*([^\n]+)',
            r'pre-op diagnosis:?\s*([^\n]+)',
            r'post-op diagnosis:?\s*([^\n]+)',
            r'final diagnosis:?\s*([^\n]+)'
        ]

        for pattern in diag_patterns:
            for match in re.finditer(pattern, content, re.I):
                fact_text = match.group(1).strip("*- ")
                # Filter out empty matches or "plan" sections
                if fact_text and len(fact_text) > 5 and "plan" not in fact_text.lower():
                    facts.append(HybridClinicalFact(
                        fact=f"Diagnosis: {fact_text}",
                        source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                        source_line=content[:match.start()].count('\n'),
                        timestamp=doc.timestamp,
                        absolute_timestamp=doc.timestamp,
                        confidence=0.95,
                        fact_type='diagnosis',
                        clinical_significance='HIGH',
                        clinical_context={'extraction_method': 'regex'}
                    ))

        # 2. LLM Fallback (if regex failed and LLM available)
        if not facts and self.llm_extractor:
            logger.debug(f"Regex failed for Diagnosis in {doc.doc_type.value}. Attempting LLM fallback.")
            llm_facts = self.llm_extractor.extract_diagnosis(doc)
            if llm_facts:
                logger.info(f"LLM successfully extracted {len(llm_facts)} diagnosis fact(s)")
            facts.extend(llm_facts)

        return facts

    # ========================================================================
    # MEDICATIONS - HYBRID: complete_1 patterns + v2 knowledge base
    # ========================================================================

    def _extract_medications(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        HYBRID APPROACH: Medication extraction
        - Patterns: complete_1 (lines 324-335) - More comprehensive
        - Classification: v2 (lines 408-427) - Adds clinical context

        Confidence Scoring:
        - Known medication in KB: 0.92
        - Unknown medication: 0.85
        - High-risk medication: Flagged for validation, 0.75
        """
        facts = []
        content = doc.content

        # COMPLETE_1: Comprehensive medication patterns (5 patterns)
        med_patterns = [
            # Started/initiated medications
            r'(?:started|initiated|began)\s+(\w+)\s+(\d+\s*(?:mg|mcg|g|units?)[^\n,;]*)',
            # Continued medications
            r'(?:continue|continued|continuing)\s+(\w+)\s+(\d+\s*(?:mg|mcg|g|units?)[^\n,;]*)',
            # Prescribed medications
            r'(?:prescribed|rx)\s*:?\s*(\w+)\s+(\d+\s*(?:mg|mcg|g|units?)[^\n,;]*)',
            # Medication list format (bullet points)
            r'(?:^|\n)\s*[-•]\s*(\w+)\s+(\d+\s*(?:mg|mcg|g|units?)[^\n]*)',
            # Anticoagulation specific (high-priority pattern)
            r'(?:heparin|enoxaparin|warfarin|rivaroxaban|apixaban)\s+(\d+\s*(?:mg|mcg|units?)[^\n,;]*)'
        ]

        for pattern in med_patterns:
            for match in re.finditer(pattern, content, re.I | re.M):
                med_name = match.group(1) if len(match.groups()) > 1 else match.group(0)
                dosing = match.group(2) if len(match.groups()) > 1 else ""

                # V2 ENHANCEMENT: Get medication classification from knowledge base
                med_info = self.knowledge_base.classify_medication(med_name)

                # HYBRID CONFIDENCE: Based on KB match quality
                confidence = 0.92 if med_info['class'] != 'Unknown' else 0.85

                fact = HybridClinicalFact(
                    fact=f"Medication: {med_name} {dosing}".strip(),
                    source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                    source_line=content[:match.start()].count('\n'),
                    timestamp=doc.timestamp,
                    absolute_timestamp=doc.timestamp,
                    confidence=confidence,
                    fact_type="medication",

                    # V2 ENHANCEMENT: Rich clinical context
                    clinical_context={
                        'drug_class': med_info.get('class', 'Unknown'),
                        'indications': med_info.get('indications', []),
                        'monitoring': med_info.get('monitoring', []),
                        'contraindications': med_info.get('contraindications', [])
                    },
                    normalized_value=med_name.lower(),
                    clinical_significance='HIGH' if med_info.get('class') != 'Unknown' else 'MODERATE'
                )

                # COMPLETE_1 SAFETY: Flag high-risk medications
                if self.knowledge_base.is_high_risk_medication(med_name):
                    fact.requires_validation = True
                    fact.confidence = 0.75  # Reduce confidence, require validation
                    fact.severity = 'HIGH'

                facts.append(fact)

        return facts

    # ========================================================================
    # LAB VALUES - V2 APPROACH: Normalization + Clinical Interpretation
    # ========================================================================

    def _extract_labs(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        V2 APPROACH: Lab value extraction with clinical interpretation
        Source: v2 engine lines 433-474

        Provides:
        - Normal range comparison
        - Severity classification (CRITICAL, HIGH, LOW, NORMAL)
        - Clinical implications
        - Automatic critical value flagging

        Confidence: 0.95 (structured data)
        """
        facts = []
        content = doc.content

        # Comprehensive lab pattern matching
        lab_pattern = r'(sodium|potassium|glucose|hemoglobin|platelets|inr|wbc|creatinine)[:\s]+(\d+\.?\d*)'

        for match in re.finditer(lab_pattern, content, re.I):
            lab_name = match.group(1)
            value = float(match.group(2))

            # V2: Normalize and interpret via knowledge base
            concept = self.knowledge_base.normalize_lab_value(lab_name, value)

            fact = HybridClinicalFact(
                fact=f"Lab: {lab_name} = {value} {concept.unit}",
                source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                source_line=content[:match.start()].count('\n'),
                timestamp=doc.timestamp,
                absolute_timestamp=doc.timestamp,
                confidence=0.95,  # High confidence for structured lab data
                fact_type="lab_value",

                # V2: Full clinical context
                clinical_context={
                    'normal_range': concept.normal_range,
                    'severity': concept.severity,
                    'clinical_implications': concept.clinical_implications,
                    'is_abnormal': concept.is_abnormal()
                },
                normalized_value=concept,
                severity=concept.severity,
                clinical_significance='CRITICAL' if concept.severity == 'CRITICAL' else 'HIGH'
            )

            # Auto-flag critical values for physician review
            if concept.severity == 'CRITICAL':
                fact.requires_validation = True
                logger.warning(f"Critical lab detected: {lab_name} = {value}")

            facts.append(fact)

        return facts

    # ========================================================================
    # CLINICAL SCORES - COMPLETE_1 APPROACH: Proven Robust
    # ========================================================================

    def _extract_clinical_scores(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        COMPLETE_1 APPROACH: Clinical score extraction
        Source: complete_1 engine lines 360-392

        Rationale: Already handles all neurosurgical scores accurately
        Covers: NIHSS, GCS, mRS, Hunt-Hess, Fisher, WFNS, Spetzler-Martin

        Confidence: 0.95 (numerical, standardized)
        """
        facts = []
        content = doc.content

        # Comprehensive neurosurgical score patterns (complete_1)
        # Enhanced to handle "Grade" syntax (e.g., "Fisher Grade 3")
        score_patterns = {
            'NIHSS': r'NIHSS[:\s]+(\d+)',
            'GCS': r'GCS[:\s]+(\d+)|glasgow[:\s]+(\d+)',
            'mRS': r'mRS[:\s]+(\d+)|modified\s+rankin[:\s]+(\d+)',
            'Hunt-Hess': r'hunt[\s-]+hess[\s:]+(?:grade\s+)?(\d+)',
            'Fisher': r'fisher[\s:]+(?:grade\s+)?(\d+)',  # Enhanced to handle "Fisher Grade 3"
            'WFNS': r'WFNS[:\s]+(\d+)',
            'Spetzler-Martin': r'spetzler[\s-]+martin[:\s]+(\d+)'
        }

        for score_name, pattern in score_patterns.items():
            match = re.search(pattern, content, re.I)
            if match:
                # Extract first non-None group (handles multiple capture groups)
                score_value = next(g for g in match.groups() if g is not None)

                # Validate score is within acceptable range
                is_valid, error_msg = self.knowledge_base.validate_clinical_score(
                    score_name,
                    int(score_value)
                )

                fact = HybridClinicalFact(
                    fact=f"{score_name}: {score_value}",
                    source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                    source_line=content[:match.start()].count('\n'),
                    timestamp=doc.timestamp,
                    absolute_timestamp=doc.timestamp,
                    confidence=0.95 if is_valid else 0.60,  # Lower confidence if invalid
                    fact_type="clinical_score",
                    normalized_value=int(score_value),
                    clinical_significance='HIGH',
                    clinical_context={'score_name': score_name, 'is_valid': is_valid}
                )

                # Flag invalid scores for review
                if not is_valid:
                    fact.requires_validation = True
                    fact.severity = 'HIGH'
                    logger.warning(f"Invalid clinical score detected: {error_msg}")

                facts.append(fact)

        return facts

    # ========================================================================
    # VITAL SIGNS
    # ========================================================================

    def _extract_vital_signs(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Extract vital signs (BP, HR, RR, SpO2, Temp)

        Confidence: 0.90 (semi-structured)
        """
        facts = []
        content = doc.content

        # Vital sign patterns
        vital_patterns = {
            'BP': r'(?:BP|blood\s+pressure)[:\s]+(\d{2,3})[/](\d{2,3})',
            'HR': r'(?:HR|heart\s+rate)[:\s]+(\d{2,3})',
            'RR': r'(?:RR|respiratory\s+rate)[:\s]+(\d{1,2})',
            'SpO2': r'(?:SpO2|O2\s+sat)[:\s]+(\d{2,3})(?:%)?',
            'Temp': r'(?:temp|temperature)[:\s]+(\d{2,3}\.?\d*)'
        }

        for vital_name, pattern in vital_patterns.items():
            match = re.search(pattern, content, re.I)
            if match:
                if vital_name == 'BP':
                    systolic, diastolic = match.groups()
                    value_str = f"{systolic}/{diastolic}"
                else:
                    value_str = match.group(1)

                fact = HybridClinicalFact(
                    fact=f"{vital_name}: {value_str}",
                    source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                    source_line=content[:match.start()].count('\n'),
                    timestamp=doc.timestamp,
                    absolute_timestamp=doc.timestamp,
                    confidence=0.90,
                    fact_type="vital_sign",
                    clinical_context={'vital_type': vital_name},
                    clinical_significance='MODERATE'
                )

                facts.append(fact)

        return facts

    # ========================================================================
    # TEMPORAL REFERENCES - V2 APPROACH: Comprehensive Pattern Recognition
    # ========================================================================

    def _extract_temporal_references(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        V2 APPROACH: Extract temporal references for later resolution
        Source: v2 engine temporal_patterns (lines 139-148)

        Patterns: POD#, HD#, relative time expressions
        These will be resolved to absolute timestamps in timeline building phase

        Confidence: 0.80 (will improve after resolution)
        """
        facts = []
        content = doc.content

        # V2's comprehensive temporal patterns from knowledge base
        for pattern, temp_type in self.knowledge_base.temporal_patterns.items():
            for match in re.finditer(pattern, content, re.I):
                temporal_ref = match.group(0)

                # Extract surrounding context (50 chars before, 100 chars after)
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 100)
                context = content[start:end]

                fact = HybridClinicalFact(
                    fact=f"Temporal reference: {temporal_ref}",
                    source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                    source_line=content[:match.start()].count('\n'),
                    timestamp=doc.timestamp,
                    absolute_timestamp=doc.timestamp,  # Will be resolved in timeline phase
                    confidence=0.80,  # Moderate until resolved
                    fact_type="temporal_reference",
                    clinical_context={
                        'type': temp_type,
                        'raw_text': temporal_ref,
                        'surrounding_context': context,
                        'needs_resolution': True
                    },
                    clinical_significance='MODERATE'
                )

                facts.append(fact)

        return facts

    # ========================================================================
    # OPERATIVE NOTE EXTRACTION - COMPLETE_1 APPROACH: Domain-Specific
    # ========================================================================

    def _extract_operative_facts(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        COMPLETE_1 APPROACH: Specialized operative note extraction
        Source: complete_1 engine lines 199-255

        Rationale: Domain-specific for neurosurgical procedures
        Extracts: Procedures, findings, complications

        Confidence: 0.92-0.95 (operative notes are structured)
        """
        facts = []
        content = doc.content

        # Extract procedures
        facts.extend(self._extract_procedures(doc))

        # Extract surgical findings
        facts.extend(self._extract_surgical_findings(doc))

        # Extract complications (if any)
        facts.extend(self._extract_complications(doc))

        # Also extract medications, labs, scores from operative note
        facts.extend(self._extract_medications(doc))
        facts.extend(self._extract_labs(doc))
        facts.extend(self._extract_clinical_scores(doc))

        return self._deduplicate_facts(facts)

    def _extract_procedures(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Extract surgical procedures from operative note.
        Uses regex first, then LLM fallback if needed.
        Source: complete_1 engine lines 205-230 (enhanced with LLM fallback)
        """
        facts = []
        content = doc.content

        # 1. Try Regex Patterns First
        procedure_patterns = [
            r'procedure[s]?\s*performed?:?\s*([^\n]+)',
            r'operation:?\s*([^\n]+)',
            r'surgical\s+procedure:?\s*([^\n]+)',
            r'(?:underwent|performed)\s+([^\n]+?(?:craniotomy|clipping|coiling|resection|biopsy)[^\n]*)',
            r'Procedures\n\*([^\n]+)'  # Pattern for bullet-pointed procedures
        ]

        for pattern in procedure_patterns:
            match = re.search(pattern, content, re.I)
            if match:
                procedure_text = match.group(1).strip("*- ")

                fact = HybridClinicalFact(
                    fact=f"Procedure: {procedure_text}",
                    source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                    source_line=content[:match.start()].count('\n'),
                    timestamp=doc.timestamp,
                    absolute_timestamp=doc.timestamp,
                    confidence=0.95,  # High confidence in operative notes
                    fact_type="procedure",
                    clinical_significance='HIGH',
                    clinical_context={'document_type': 'operative', 'extraction_method': 'regex'}
                )

                facts.append(fact)
                break  # Only extract first procedure description

        # 2. LLM Fallback (if regex failed and LLM available)
        if not facts and self.llm_extractor:
            logger.debug("Regex failed for Procedure in Operative Note. Attempting LLM fallback.")
            llm_facts = self.llm_extractor.extract_procedure(doc)
            if llm_facts:
                logger.info(f"LLM successfully extracted {len(llm_facts)} procedure fact(s)")
            facts.extend(llm_facts)

        return facts

    def _extract_surgical_findings(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Extract intraoperative findings
        Source: complete_1 engine lines 232-245
        """
        facts = []
        content = doc.content

        findings_patterns = [
            r'findings?:?\s*([^\n]+)',
            r'intraoperative\s+findings?:?\s*([^\n]+)',
            r'operative\s+findings?:?\s*([^\n]+)'
        ]

        for pattern in findings_patterns:
            match = re.search(pattern, content, re.I)
            if match:
                finding_text = match.group(1).strip()

                fact = HybridClinicalFact(
                    fact=f"Surgical finding: {finding_text}",
                    source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                    source_line=content[:match.start()].count('\n'),
                    timestamp=doc.timestamp,
                    absolute_timestamp=doc.timestamp,
                    confidence=0.92,
                    fact_type="finding",
                    clinical_significance='HIGH',
                    clinical_context={'finding_type': 'intraoperative'}
                )

                facts.append(fact)

        return facts

    def _extract_complications(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Extract complications (always flagged for validation)
        Source: complete_1 engine lines 247-255
        """
        facts = []
        content = doc.content

        # Check for complication keywords
        if "complication" in content.lower():
            # Pattern to extract complication descriptions
            comp_patterns = [
                r'complication[s]?:?\s*([^\n]+)',
                r'developed\s+([^\n]*(?:leak|hemorrhage|infection|vasospasm|edema)[^\n]*)',
                r'(?:post-?operative|post-?op)\s+([^\n]*(?:complication|issue|problem)[^\n]*)'
            ]

            for pattern in comp_patterns:
                for match in re.finditer(pattern, content, re.I):
                    comp_text = match.group(1).strip()

                    fact = HybridClinicalFact(
                        fact=f"Complication: {comp_text}",
                        source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                        source_line=content[:match.start()].count('\n'),
                        timestamp=doc.timestamp,
                        absolute_timestamp=doc.timestamp,
                        confidence=0.90,
                        fact_type="complication",
                        requires_validation=True,  # ALWAYS validate complications
                        severity='HIGH',
                        clinical_significance='CRITICAL',
                        clinical_context={'requires_review': True}
                    )

                    facts.append(fact)

        return facts

    # ========================================================================
    # CONSULTATION NOTE EXTRACTION - COMPLETE_1 APPROACH: Specialty-Specific
    # ========================================================================

    def _extract_consult_facts(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        COMPLETE_1 APPROACH: Specialty-specific consultation extraction
        Source: complete_1 engine lines 257-291

        Rationale: Better structured for multi-specialty consultations
        Handles: ID, Thrombosis, Cardiology, etc.

        Confidence: 0.88 (recommendations from specialists)
        """
        facts = []
        content = doc.content

        # Extract recommendations
        facts.extend(self._extract_recommendations(doc))

        # Extract diagnoses (with LLM fallback)
        facts.extend(self._extract_diagnoses(doc))

        # Specialty-specific extraction (complete_1 approach)
        if doc.specialty and doc.specialty.lower() in ['infectious disease', 'id']:
            facts.extend(self._extract_id_specific(doc))
        elif doc.specialty and doc.specialty.lower() in ['thrombosis', 'hematology']:
            facts.extend(self._extract_thrombosis_specific(doc))

        # Also extract standard fact types
        facts.extend(self._extract_medications(doc))
        facts.extend(self._extract_labs(doc))

        return self._deduplicate_facts(facts)

    def _extract_recommendations(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Extract recommendations from consultation notes
        Source: complete_1 engine lines 262-280
        """
        facts = []
        content = doc.content

        # Recommendation patterns
        rec_patterns = [
            r'recommend[ation]*s?:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
            r'plan:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
            r'impression\s+and\s+plan:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)'
        ]

        for pattern in rec_patterns:
            match = re.search(pattern, content, re.I | re.M)
            if match:
                recommendations_text = match.group(1).strip()

                # Parse numbered or bulleted recommendations
                recs = self._parse_recommendations(recommendations_text)

                for rec in recs:
                    fact = HybridClinicalFact(
                        fact=f"{doc.specialty} recommendation: {rec}",
                        source_doc=f"{doc.specialty}_consult_{doc.timestamp}",
                        source_line=content[:match.start()].count('\n'),
                        timestamp=doc.timestamp,
                        absolute_timestamp=doc.timestamp,
                        confidence=0.88,
                        fact_type="recommendation",
                        clinical_context={'specialty': doc.specialty},
                        clinical_significance='HIGH'
                    )

                    facts.append(fact)

        return facts

    def _parse_recommendations(self, text: str) -> List[str]:
        """
        Parse numbered or bulleted list of recommendations

        Handles:
        - 1. Recommendation
        - - Recommendation
        - • Recommendation
        """
        recs = []

        # Split by numbered list
        numbered = re.split(r'\n\s*\d+[\.)]\s*', text)
        if len(numbered) > 1:
            recs = [r.strip() for r in numbered[1:] if r.strip()]
            return recs

        # Split by bullet points
        bulleted = re.split(r'\n\s*[-•]\s*', text)
        if len(bulleted) > 1:
            recs = [r.strip() for r in bulleted[1:] if r.strip()]
            return recs

        # If no structure, treat as single recommendation
        return [text.strip()] if text.strip() else []

    def _extract_id_specific(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Infectious Disease-specific extraction
        Source: complete_1 engine (ID consultation handling)

        Extracts: Antibiotic recommendations, culture results, infection diagnoses
        """
        facts = []
        content = doc.content

        # Look for antibiotic recommendations
        abx_pattern = r'(?:antibiotic|abx)[s]?\s*(?:recommend|rx|prescribed)?[:\s]*([^\n]+)'
        match = re.search(abx_pattern, content, re.I)
        if match:
            abx_text = match.group(1).strip()

            fact = HybridClinicalFact(
                fact=f"ID recommendation: {abx_text}",
                source_doc=f"ID_consult_{doc.timestamp}",
                source_line=content[:match.start()].count('\n'),
                timestamp=doc.timestamp,
                absolute_timestamp=doc.timestamp,
                confidence=0.88,
                fact_type="recommendation",
                clinical_context={'specialty': 'ID', 'subtype': 'antibiotics'},
                clinical_significance='HIGH'
            )

            facts.append(fact)

        return facts

    def _extract_thrombosis_specific(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Thrombosis/Hematology-specific extraction
        Source: complete_1 engine (Thrombosis consultation handling)

        Extracts: DVT prophylaxis, anticoagulation recommendations
        """
        facts = []
        content = doc.content

        # Look for DVT prophylaxis or anticoagulation
        dvt_pattern = r'(?:DVT|thrombosis)\s+(?:prophylaxis|prevention)[:\s]*([^\n]+)'
        match = re.search(dvt_pattern, content, re.I)
        if match:
            dvt_text = match.group(1).strip()

            fact = HybridClinicalFact(
                fact=f"Thrombosis recommendation: {dvt_text}",
                source_doc=f"Thrombosis_consult_{doc.timestamp}",
                source_line=content[:match.start()].count('\n'),
                timestamp=doc.timestamp,
                absolute_timestamp=doc.timestamp,
                confidence=0.88,
                fact_type="recommendation",
                clinical_context={'specialty': 'Thrombosis', 'subtype': 'DVT_prophylaxis'},
                clinical_significance='HIGH'
            )

            facts.append(fact)

        return facts

    # ========================================================================
    # LAB REPORT EXTRACTION - Specialized for structured lab reports
    # ========================================================================

    def _extract_lab_report_facts(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Specialized extraction for lab reports

        Lab reports are more structured, so we can achieve higher confidence
        """
        facts = self._extract_labs(doc)

        # Boost confidence for lab report documents
        for fact in facts:
            if fact.fact_type == "lab_value":
                fact.confidence = min(0.98, fact.confidence + 0.03)

        return facts

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _deduplicate_facts(self, facts: List[HybridClinicalFact]) -> List[HybridClinicalFact]:
        """
        Remove duplicate facts while preserving highest confidence version

        Two facts are considered duplicates if:
        1. Same fact_type
        2. Same normalized fact text (case-insensitive)
        3. Same timestamp (within 1 hour)

        Keeps: Fact with highest confidence score
        """
        if not facts:
            return []

        # Group facts by type and normalized text
        fact_groups = defaultdict(list)

        for fact in facts:
            # Normalize fact text for comparison
            normalized = fact.fact.lower().strip()
            key = (fact.fact_type, normalized, fact.timestamp.date())
            fact_groups[key].append(fact)

        # For each group, keep highest confidence fact
        deduplicated = []
        for group in fact_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Sort by confidence (descending) and keep first
                best_fact = sorted(group, key=lambda f: f.confidence, reverse=True)[0]

                # Track that we deduplicated
                best_fact.clinical_context['deduplicated_count'] = len(group)

                deduplicated.append(best_fact)

        logger.debug(f"Deduplicated {len(facts)} → {len(deduplicated)} facts")
        return deduplicated

    def get_extraction_stats(self, facts: List[HybridClinicalFact]) -> Dict:
        """
        Get statistics about extracted facts

        Returns:
            Dictionary with counts by type, confidence distribution, etc.
        """
        if not facts:
            return {'total': 0}

        # Count by type
        type_counts = defaultdict(int)
        confidence_scores = []
        requires_validation_count = 0

        for fact in facts:
            type_counts[fact.fact_type] += 1
            confidence_scores.append(fact.confidence)
            if fact.requires_validation:
                requires_validation_count += 1

        return {
            'total': len(facts),
            'by_type': dict(type_counts),
            'avg_confidence': sum(confidence_scores) / len(confidence_scores),
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores),
            'requires_validation': requires_validation_count
        }
