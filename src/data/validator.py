"""
Claim validation utilities for ensuring transformed prompts meet quality standards.

This module provides comprehensive validation logic to ensure that transformed
claims are debatable, appropriate, and suitable for persuasive dialogue.
"""

import re
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a claim."""
    rule_name: str
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for a claim."""
    claim: str
    is_valid: bool
    overall_score: float
    issues: List[ValidationIssue]
    passed_rules: List[str]
    
    @property
    def has_errors(self) -> bool:
        """Check if report contains any error-level issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if report contains any warning-level issues."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)


class ClaimValidator:
    """
    Validates transformed claims for quality and appropriateness.
    
    This class implements comprehensive validation rules to ensure that
    transformed claims meet the standards for persuasive dialogue research.
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.quality_thresholds = self._initialize_thresholds()
        
    def _initialize_validation_rules(self) -> Dict[str, Dict]:
        """Initialize validation rules with their configurations."""
        return {
            "debatable_stance": {
                "weight": 0.25,
                "severity": ValidationSeverity.ERROR,
                "description": "Claim must present a debatable position"
            },
            "appropriate_content": {
                "weight": 0.30,
                "severity": ValidationSeverity.CRITICAL,
                "description": "Claim must be appropriate for academic discussion"
            },
            "clear_position": {
                "weight": 0.20,
                "severity": ValidationSeverity.ERROR,
                "description": "Claim must have a clear, understandable position"
            },
            "grammatical_quality": {
                "weight": 0.10,
                "severity": ValidationSeverity.WARNING,
                "description": "Claim should be grammatically correct"
            },
            "length_appropriateness": {
                "weight": 0.05,
                "severity": ValidationSeverity.WARNING,
                "description": "Claim should be appropriate length"
            },
            "context_relevance": {
                "weight": 0.10,
                "severity": ValidationSeverity.INFO,
                "description": "Claim should maintain relevant context"
            }
        }
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds for validation."""
        return {
            "minimum_score": 0.7,
            "warning_score": 0.8,
            "excellent_score": 0.9,
            "min_word_count": 5,
            "max_word_count": 100,
            "min_char_count": 20,
            "max_char_count": 500
        }
    
    def validate_claim(self, claim: str, original_prompt: Optional[str] = None) -> ValidationReport:
        """
        Validate a transformed claim comprehensively.
        
        Args:
            claim: The transformed claim to validate
            original_prompt: Optional original prompt for context validation
            
        Returns:
            ValidationReport containing detailed validation results
        """
        logger.info(f"Validating claim: {claim[:50]}...")
        
        issues = []
        passed_rules = []
        total_score = 0.0
        
        for rule_name, rule_config in self.validation_rules.items():
            try:
                rule_result = self._apply_validation_rule(rule_name, claim, original_prompt)
                
                if rule_result["passed"]:
                    passed_rules.append(rule_name)
                    total_score += rule_config["weight"]
                else:
                    issue = ValidationIssue(
                        rule_name=rule_name,
                        severity=rule_config["severity"],
                        message=rule_result["message"],
                        suggestion=rule_result.get("suggestion")
                    )
                    issues.append(issue)
                    
            except Exception as e:
                logger.error(f"Validation rule {rule_name} failed: {e}")
                issue = ValidationIssue(
                    rule_name=rule_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule failed: {str(e)}"
                )
                issues.append(issue)
        
        # Determine overall validity
        has_critical_issues = any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        has_error_issues = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        is_valid = (total_score >= self.quality_thresholds["minimum_score"] and 
                   not has_critical_issues and not has_error_issues)
        
        report = ValidationReport(
            claim=claim,
            is_valid=is_valid,
            overall_score=total_score,
            issues=issues,
            passed_rules=passed_rules
        )
        
        logger.info(f"Validation complete. Valid: {is_valid}, Score: {total_score:.2f}")
        return report
    
    def _apply_validation_rule(self, rule_name: str, claim: str, original_prompt: Optional[str]) -> Dict:
        """Apply a specific validation rule to the claim."""
        
        if rule_name == "debatable_stance":
            return self._validate_debatable_stance(claim)
        elif rule_name == "appropriate_content":
            return self._validate_appropriate_content(claim)
        elif rule_name == "clear_position":
            return self._validate_clear_position(claim)
        elif rule_name == "grammatical_quality":
            return self._validate_grammatical_quality(claim)
        elif rule_name == "length_appropriateness":
            return self._validate_length_appropriateness(claim)
        elif rule_name == "context_relevance":
            return self._validate_context_relevance(claim, original_prompt)
        else:
            raise ValueError(f"Unknown validation rule: {rule_name}")
    
    def _validate_debatable_stance(self, claim: str) -> Dict:
        """Validate that the claim presents a debatable stance."""
        claim_lower = claim.lower()
        
        # Check for absolute statements that are not debatable
        absolute_indicators = [
            'always', 'never', 'all', 'none', 'every', 'no one', 'everyone',
            'impossible', 'certain', 'definitely', 'absolutely', 'completely'
        ]
        
        for indicator in absolute_indicators:
            if f' {indicator} ' in f' {claim_lower} ':
                return {
                    "passed": False,
                    "message": f"Contains absolute statement: '{indicator}'",
                    "suggestion": "Rephrase to allow for debate and different perspectives"
                }
        
        # Check for opinion/stance indicators
        stance_indicators = [
            'should', 'could', 'might', 'may', 'consider', 'debate', 'argument',
            'position', 'believe', 'think', 'suggest', 'propose', 'claim'
        ]
        
        has_stance = any(indicator in claim_lower for indicator in stance_indicators)
        
        if not has_stance:
            return {
                "passed": False,
                "message": "Lacks clear debatable position or stance indicators",
                "suggestion": "Add words like 'should', 'could', or 'consider' to indicate a debatable position"
            }
        
        return {
            "passed": True,
            "message": "Contains appropriate debatable stance"
        }
    
    def _validate_appropriate_content(self, claim: str) -> Dict:
        """Validate that the claim is appropriate for academic discussion."""
        claim_lower = claim.lower()
        
        # Check for explicit harmful content
        harmful_terms = [
            'kill', 'murder', 'bomb', 'attack', 'destroy', 'harm', 'hurt',
            'violence', 'weapon', 'explosive', 'poison', 'drug', 'illegal'
        ]
        
        for term in harmful_terms:
            if term in claim_lower:
                return {
                    "passed": False,
                    "message": f"Contains potentially harmful term: '{term}'",
                    "suggestion": "Rephrase to discuss the topic in an academic context"
                }
        
        # Check for inappropriate language
        inappropriate_terms = [
            'stupid', 'idiot', 'moron', 'dumb', 'retarded', 'crazy',
            'insane', 'psycho', 'freak', 'loser'
        ]
        
        for term in inappropriate_terms:
            if term in claim_lower:
                return {
                    "passed": False,
                    "message": f"Contains inappropriate language: '{term}'",
                    "suggestion": "Use more professional and respectful language"
                }
        
        return {
            "passed": True,
            "message": "Content is appropriate for academic discussion"
        }
    
    def _validate_clear_position(self, claim: str) -> Dict:
        """Validate that the claim has a clear, understandable position."""
        
        # Check minimum length for meaningful content
        if len(claim.strip()) < self.quality_thresholds["min_char_count"]:
            return {
                "passed": False,
                "message": "Claim too short to contain clear position",
                "suggestion": f"Expand to at least {self.quality_thresholds['min_char_count']} characters"
            }
        
        # Check for complete sentences
        if not claim.strip().endswith(('.', '!', '?')):
            return {
                "passed": False,
                "message": "Claim should end with proper punctuation",
                "suggestion": "Add appropriate punctuation to complete the sentence"
            }
        
        # Check for subject-verb structure
        words = claim.split()
        if len(words) < self.quality_thresholds["min_word_count"]:
            return {
                "passed": False,
                "message": f"Claim too short (minimum {self.quality_thresholds['min_word_count']} words)",
                "suggestion": "Expand the claim to provide more context and clarity"
            }
        
        return {
            "passed": True,
            "message": "Claim has clear, understandable position"
        }
    
    def _validate_grammatical_quality(self, claim: str) -> Dict:
        """Validate basic grammatical quality of the claim."""
        
        # Check for double spaces or formatting issues first
        if '  ' in claim or claim != claim.strip():
            return {
                "passed": False,
                "message": "Contains formatting issues (extra spaces)",
                "suggestion": "Clean up spacing and formatting"
            }
        
        # Check for basic capitalization
        if not claim[0].isupper():
            return {
                "passed": False,
                "message": "Claim should start with capital letter",
                "suggestion": "Capitalize the first letter of the claim"
            }
        
        # Check for excessive repetition
        words = claim.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only check meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > 2]
        if repeated_words:
            return {
                "passed": False,
                "message": f"Excessive word repetition: {', '.join(repeated_words)}",
                "suggestion": "Vary word choice to improve readability"
            }
        
        return {
            "passed": True,
            "message": "Basic grammatical quality is acceptable"
        }
    
    def _validate_length_appropriateness(self, claim: str) -> Dict:
        """Validate that the claim length is appropriate."""
        
        word_count = len(claim.split())
        char_count = len(claim)
        
        if word_count > self.quality_thresholds["max_word_count"]:
            return {
                "passed": False,
                "message": f"Claim too long ({word_count} words, max {self.quality_thresholds['max_word_count']})",
                "suggestion": "Shorten the claim while maintaining the core argument"
            }
        
        if char_count > self.quality_thresholds["max_char_count"]:
            return {
                "passed": False,
                "message": f"Claim too long ({char_count} characters, max {self.quality_thresholds['max_char_count']})",
                "suggestion": "Reduce length while preserving meaning"
            }
        
        return {
            "passed": True,
            "message": "Claim length is appropriate"
        }
    
    def _validate_context_relevance(self, claim: str, original_prompt: Optional[str]) -> Dict:
        """Validate that the claim maintains relevant context from original."""
        
        if not original_prompt:
            return {
                "passed": True,
                "message": "No original prompt provided for context validation"
            }
        
        # Extract meaningful words from both texts
        claim_words = set(re.findall(r'\b\w{3,}\b', claim.lower()))
        original_words = set(re.findall(r'\b\w{3,}\b', original_prompt.lower()))
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        claim_content = claim_words - stop_words
        original_content = original_words - stop_words
        
        if not original_content:
            return {
                "passed": True,
                "message": "Original prompt has no meaningful content words"
            }
        
        # Calculate overlap
        overlap = len(claim_content & original_content)
        relevance_ratio = overlap / len(original_content)
        
        if relevance_ratio < 0.1:
            return {
                "passed": False,
                "message": f"Low context relevance ({relevance_ratio:.1%} overlap)",
                "suggestion": "Include more relevant terms from the original context"
            }
        
        return {
            "passed": True,
            "message": f"Good context relevance ({relevance_ratio:.1%} overlap)"
        }
    
    def batch_validate(self, claims: List[str], original_prompts: Optional[List[str]] = None) -> List[ValidationReport]:
        """
        Validate multiple claims in batch.
        
        Args:
            claims: List of claims to validate
            original_prompts: Optional list of original prompts for context
            
        Returns:
            List of ValidationReport objects
        """
        if original_prompts and len(original_prompts) != len(claims):
            raise ValueError("Original prompts list must match claims list length")
        
        reports = []
        for i, claim in enumerate(claims):
            original = original_prompts[i] if original_prompts else None
            report = self.validate_claim(claim, original)
            reports.append(report)
        
        logger.info(f"Batch validation complete. {len(reports)} claims processed.")
        return reports
    
    def get_validation_summary(self, reports: List[ValidationReport]) -> Dict:
        """
        Generate summary statistics for validation reports.
        
        Args:
            reports: List of validation reports
            
        Returns:
            Dictionary containing validation statistics
        """
        if not reports:
            return {}
        
        total = len(reports)
        valid_count = sum(1 for r in reports if r.is_valid)
        
        # Score statistics
        scores = [r.overall_score for r in reports]
        avg_score = sum(scores) / total
        min_score = min(scores)
        max_score = max(scores)
        
        # Issue statistics
        all_issues = [issue for report in reports for issue in report.issues]
        issue_counts = {}
        severity_counts = {}
        
        for issue in all_issues:
            issue_counts[issue.rule_name] = issue_counts.get(issue.rule_name, 0) + 1
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
        
        return {
            "total_claims": total,
            "valid_claims": valid_count,
            "validation_rate": valid_count / total,
            "average_score": avg_score,
            "score_range": {"min": min_score, "max": max_score},
            "common_issues": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)),
            "severity_distribution": severity_counts,
            "total_issues": len(all_issues)
        }