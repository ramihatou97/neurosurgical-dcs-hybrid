"""
LLM-Powered Extractor Module (Smart Extractor)

This module provides targeted LLM calls to extract specific, complex
facts that are difficult to capture with regex.

It is used as a fallback by the main HybridFactExtractor.
"""
import logging
from typing import List, Optional
from anthropic import Anthropic, AnthropicError

from src.core.data_models import HybridClinicalFact, ClinicalDocument, FactType, DocumentType
from datetime import datetime

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "claude-3-haiku-20240307"

class LlmExtractor:

    def __init__(self, client: Optional[Anthropic]):
        """
        Initializes the LLM Extractor with a pre-configured Anthropic client.

        Args:
            client: An initialized Anthropic client (or None).
        """
        self.client = client

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> Optional[str]:
        """Helper function to make a targeted call to the LLM."""
        if not self.client:
            logger.warning("LLM client not available. Skipping LLM extraction fallback.")
            return None

        try:
            message = self.client.messages.create(
                model=EXTRACTION_MODEL,
                max_tokens=max_tokens,
                temperature=0.0, # Factual extraction
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            response_text = message.content[0].text.strip()
            if response_text.lower() == "none":
                return None
            return response_text
        except AnthropicError as e:
            logger.error(f"LLM extraction API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected LLM extraction error: {e}")
            return None

    def extract_diagnosis(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Uses LLM to find "Diagnosis" or "Assessment".
        """
        system = (
            "You are a medical data extractor. Your task is to extract the 'Diagnosis' or 'Assessment' from the provided clinical note. "
            "Return *only* the diagnosis text, separated by newlines if multiple. If no diagnosis is found, return 'None'."
        )
        user = f"Here is the document content:\n\n{doc.content}"

        result_text = self._call_llm(system, user)
        facts = []

        if result_text:
            for diag in result_text.split('\n'):
                diag_clean = diag.strip("*- ")
                if diag_clean:
                    facts.append(HybridClinicalFact(
                        fact=f"Diagnosis: {diag_clean}",
                        source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                        source_line=0, # Line number is ambiguous with LLM
                        timestamp=doc.timestamp,
                        absolute_timestamp=doc.timestamp,
                        confidence=0.85, # High, but less than regex
                        fact_type=FactType.DIAGNOSIS.value,
                        clinical_significance='HIGH',
                        clinical_context={'extraction_method': 'llm_fallback'}
                    ))
        return facts

    def extract_procedure(self, doc: ClinicalDocument) -> List[HybridClinicalFact]:
        """
        Uses LLM to find "Procedure Performed".
        """
        system = (
            "You are a medical data extractor. Your task is to extract the 'Procedure Performed' from the provided operative note. "
            "List multiple procedures if present, separated by newlines. Only return the procedure names. If no procedure is found, return 'None'."
        )
        user = f"Here is the document content:\n\n{doc.content}"

        result_text = self._call_llm(system, user)
        facts = []

        if result_text:
            for proc in result_text.split('\n'):
                proc_clean = proc.strip("*- ")
                if proc_clean:
                    facts.append(HybridClinicalFact(
                        fact=f"Procedure: {proc_clean}",
                        source_doc=f"{doc.doc_type.value}_{doc.timestamp}",
                        source_line=0,
                        timestamp=doc.timestamp,
                        absolute_timestamp=doc.timestamp,
                        confidence=0.85,
                        fact_type=FactType.PROCEDURE.value,
                        clinical_significance='HIGH',
                        clinical_context={'extraction_method': 'llm_fallback'}
                    ))
            return facts
        return []
