"""
Narrative Generation & LLM Client Module

This module integrates with the Anthropic API to:
1. Generate rich, narrative discharge summaries.
2. Provide a central LLM client for other modules (like the Smart Extractor).

It is called by the main engine.
"""

import os
import logging
from dotenv import load_dotenv
from anthropic import Anthropic, AnthropicError
from typing import List, Dict, Optional, Tuple

# Import the core data models
from src.core.data_models import ClinicalTimeline, HybridClinicalFact, ClinicalUncertainty

# Load environment variables (specifically ANTHROPIC_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)

# Use a cost-effective and fast model
NARRATIVE_MODEL = "claude-3-haiku-20240307"
EXTRACTION_MODEL = "claude-3-haiku-20240307"


class NarrativeGenerator:
    """
    Generates a narrative discharge summary and provides a client for LLM extraction.
    """

    def __init__(self, model_name: str = NARRATIVE_MODEL):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model_name
        self.client: Optional[Anthropic] = None

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment.")
            logger.warning("NarrativeGenerator and Smart Extractor will be disabled.")
        else:
            try:
                self.client = Anthropic(api_key=self.api_key)
                logger.info(f"NarrativeGenerator initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None

    def get_client(self) -> Optional[Anthropic]:
        """Returns the initialized Anthropic client for other modules to use."""
        return self.client

    def generate_summary(
        self,
        timeline: ClinicalTimeline,
        facts: List[HybridClinicalFact],
        uncertainties: List[ClinicalUncertainty]
    ) -> str:
        """
        Generates the full narrative summary.
        """
        if not self.client:
            logger.warning("Cannot generate narrative: LLM client not initialized.")
            return self._generate_placeholder_summary(timeline) # Fallback to basic summary

        system_prompt, user_prompt = self._build_prompt(timeline, facts, uncertainties)

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                temperature=0.2,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            narrative_text = message.content[0].text
            logger.info(f"Narrative summary generated successfully ({len(narrative_text)} chars).")
            return narrative_text

        except AnthropicError as e:
            logger.error(f"Anthropic API error during narrative generation: {e}")
            return f"[Narrative generation failed: {e}]\n\n" + self._generate_placeholder_summary(timeline)
        except Exception as e:
            logger.error(f"Unexpected error during narrative generation: {e}")
            return f"[Narrative generation failed: {e}]\n\n" + self._generate_placeholder_summary(timeline)

    def _build_prompt(
        self,
        timeline: ClinicalTimeline,
        facts: List[HybridClinicalFact],
        uncertainties: List[ClinicalUncertainty]
    ) -> Tuple[str, str]:
        """
        Builds the system and user prompts for the LLM.
        """

        system_prompt = (
            "You are a medical scribe for a senior neurosurgeon. "
            "Your task is to write a formal, clear, and concise discharge summary. "
            "Use the provided structured data. Be medically accurate. "
            "Organize the output into standard sections (e.g., Hospital Course, Discharge Medications, Follow-up)."
            "Crucially, you MUST highlight any unresolved 'HIGH' severity uncertainties."
        )

        # Build the user prompt with serialized data
        prompt_parts = []
        prompt_parts.append("Generate a discharge summary based on the following structured data:\n")

        # --- Key Info ---
        prompt_parts.append("## Key Timeline & Events")
        if timeline.admission_date:
            prompt_parts.append(f"- Admission Date: {timeline.admission_date.strftime('%Y-%m-%d')}")
        if timeline.discharge_date:
            prompt_parts.append(f"- Discharge Date: {timeline.discharge_date.strftime('%Y-%m-%d')}")
        if timeline.total_hospital_days:
            prompt_parts.append(f"- Length of Stay: {timeline.total_hospital_days} days")

        for event in timeline.key_events:
            prompt_parts.append(f"- Key Event ({event['date']}): {event['description']}")

        # --- Progression ---
        prompt_parts.append("\n## Clinical Progression")
        for key, trend_list in timeline.progression.items():
            if trend_list:
                prompt_parts.append(f"### {key.capitalize()} Progression:")
                for item in trend_list:
                    prompt_parts.append(f"- {item.get('metric', item.get('lab'))}: {item.get('trend', 'N/A')}")

        # --- Uncertainties ---
        high_severity_uncertainties = [u for u in uncertainties if u.severity == "HIGH"]
        if high_severity_uncertainties:
            prompt_parts.append("\n## CRITICAL: Unresolved Issues to Highlight")
            prompt_parts.append("The summary *must* mention the following unresolved high-severity issues:")
            for u in high_severity_uncertainties:
                prompt_parts.append(f"- {u.description} (Source: {u.conflicting_sources[0]})")

        # --- Key Facts for Reference ---
        prompt_parts.append("\n## Key Facts for Summary")

        diagnoses = [f.fact for f in facts if f.fact_type == 'diagnosis']
        if diagnoses:
            prompt_parts.append(f"Diagnosis: {'; '.join(diagnoses)}")

        procedures = [f.fact for f in facts if f.fact_type == 'procedure']
        if procedures:
            prompt_parts.append(f"Procedures: {'; '.join(procedures)}")

        meds = [f.fact for f in facts if f.fact_type == 'medication' and 'discharge' in f.source_doc.lower()]
        if meds:
            prompt_parts.append("Discharge Medications:\n" + "\n".join(f"- {m}" for m in meds))

        user_prompt = "\n".join(prompt_parts)

        return system_prompt, user_prompt

    def _generate_placeholder_summary(self, timeline: ClinicalTimeline) -> str:
        """
        Fallback function if LLM fails.
        """
        logger.warning("Falling back to placeholder summary text.")
        lines = []
        lines.append("DISCHARGE SUMMARY - [PLACEHOLDER]")
        lines.append("=" * 60)

        if timeline.admission_date:
            lines.append(f"Admission Date: {timeline.admission_date.strftime('%B %d, %Y')}")
        if timeline.discharge_date:
            lines.append(f"Discharge Date: {timeline.discharge_date.strftime('%B %d, %Y')}")
        if timeline.total_hospital_days:
            lines.append(f"Length of Stay: {timeline.total_hospital_days} days")

        lines.append("\nKey Events:")
        for event in timeline.key_events:
            lines.append(f"- {event['date']}: {event['description']}")

        lines.append("\n[Narrative generation failed or is not enabled. This is a structured data summary.]")
        return "\n".join(lines)
