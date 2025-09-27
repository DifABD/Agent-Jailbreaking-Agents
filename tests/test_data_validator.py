"""
Unit tests for claim validation utilities.

Tests the ClaimValidator class for accuracy and consistency in validating
transformed claims for quality and appropriateness.
"""

import pytest
from src.data.validator import (
    ClaimValidator,
    ValidationReport,
    ValidationIssue,
    ValidationSeverity
)


class TestClaimValidator:
    """Test cases for ClaimValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a ClaimValidator instance for testing."""
        return ClaimValidator()
    
    def test_validator_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator is not None
        assert hasattr(validator, 'validation_rules')
        assert hasattr(validator, 'quality_thresholds')
        assert len(validator.validation_rules) > 0
        assert len(validator.quality_thresholds) > 0
    
    def test_validate_claim_basic_valid(self, validator):
        """Test validation of a basic valid claim."""
        claim = "The position that cybersecurity research should be supported in academic contexts."
        
        report = validator.validate_claim(claim)
        
        assert isinstance(report, ValidationReport)
        assert report.claim == claim
        assert isinstance(report.is_valid, bool)
        assert isinstance(report.overall_score, float)
        assert isinstance(report.issues, list)
        assert isinstance(report.passed_rules, list)
        assert 0 <= report.overall_score <= 1
    
    def test_validate_claim_basic_invalid(self, validator):
        """Test validation of a basic invalid claim."""
        claim = "kill everyone"  # Too short, inappropriate, no stance
        
        report = validator.validate_claim(claim)
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
        assert len(report.issues) > 0
        assert report.overall_score < validator.quality_thresholds["minimum_score"]
    
    def test_validate_debatable_stance_valid(self, validator):
        """Test validation passes for debatable stance."""
        claim = "The argument that privacy should be balanced with security deserves consideration."
        
        result = validator._validate_debatable_stance(claim)
        
        assert result["passed"] is True
        assert "debatable" in result["message"].lower()
    
    def test_validate_debatable_stance_absolute_statement(self, validator):
        """Test validation fails for absolute statements."""
        claim = "Privacy is always more important than security in every situation."
        
        result = validator._validate_debatable_stance(claim)
        
        assert result["passed"] is False
        assert "absolute" in result["message"].lower()
        assert "always" in result["message"] or "every" in result["message"]
    
    def test_validate_debatable_stance_no_stance(self, validator):
        """Test validation fails when no clear stance is present."""
        claim = "Privacy and security are concepts that exist in technology."
        
        result = validator._validate_debatable_stance(claim)
        
        assert result["passed"] is False
        assert "stance" in result["message"].lower()
    
    def test_validate_appropriate_content_valid(self, validator):
        """Test validation passes for appropriate content."""
        claim = "The position that educational reform should be prioritized deserves consideration."
        
        result = validator._validate_appropriate_content(claim)
        
        assert result["passed"] is True
        assert "appropriate" in result["message"].lower()
    
    def test_validate_appropriate_content_harmful_terms(self, validator):
        """Test validation fails for harmful terms."""
        harmful_claims = [
            "The position that we should kill our enemies is justified.",
            "Instructions for making bombs should be available.",
            "Violence is the answer to political problems."
        ]
        
        for claim in harmful_claims:
            result = validator._validate_appropriate_content(claim)
            assert result["passed"] is False
            assert "harmful" in result["message"].lower()
    
    def test_validate_appropriate_content_inappropriate_language(self, validator):
        """Test validation fails for inappropriate language."""
        claim = "The stupid position that idiots take on this issue is moronic."
        
        result = validator._validate_appropriate_content(claim)
        
        assert result["passed"] is False
        assert "inappropriate" in result["message"].lower()
    
    def test_validate_clear_position_valid(self, validator):
        """Test validation passes for clear position."""
        claim = "The argument that renewable energy should be prioritized over fossil fuels is compelling."
        
        result = validator._validate_clear_position(claim)
        
        assert result["passed"] is True
        assert "clear" in result["message"].lower()
    
    def test_validate_clear_position_too_short(self, validator):
        """Test validation fails for too short claims."""
        claim = "Yes."
        
        result = validator._validate_clear_position(claim)
        
        assert result["passed"] is False
        assert "short" in result["message"].lower()
    
    def test_validate_clear_position_no_punctuation(self, validator):
        """Test validation fails for missing punctuation."""
        claim = "The position that education should be reformed"
        
        result = validator._validate_clear_position(claim)
        
        assert result["passed"] is False
        assert "punctuation" in result["message"].lower()
    
    def test_validate_clear_position_too_few_words(self, validator):
        """Test validation fails for too few words."""
        claim = "Education good."
        
        result = validator._validate_clear_position(claim)
        
        assert result["passed"] is False
        assert "short" in result["message"].lower()
    
    def test_validate_grammatical_quality_valid(self, validator):
        """Test validation passes for good grammatical quality."""
        claim = "The position that environmental protection should be prioritized is well-founded."
        
        result = validator._validate_grammatical_quality(claim)
        
        assert result["passed"] is True
        assert "acceptable" in result["message"].lower()
    
    def test_validate_grammatical_quality_no_capitalization(self, validator):
        """Test validation fails for missing capitalization."""
        claim = "the position that education should be reformed is valid."
        
        result = validator._validate_grammatical_quality(claim)
        
        assert result["passed"] is False
        assert "capital" in result["message"].lower()
    
    def test_validate_grammatical_quality_repetition(self, validator):
        """Test validation fails for excessive word repetition."""
        claim = "The position position position that education education should be reformed is valid."
        
        result = validator._validate_grammatical_quality(claim)
        
        assert result["passed"] is False
        assert "repetition" in result["message"].lower()
    
    def test_validate_grammatical_quality_formatting_issues(self, validator):
        """Test validation fails for formatting issues."""
        claim = "  The position that  education should be  reformed.  "
        
        result = validator._validate_grammatical_quality(claim)
        
        assert result["passed"] is False
        assert "formatting" in result["message"].lower()
    
    def test_validate_length_appropriateness_valid(self, validator):
        """Test validation passes for appropriate length."""
        claim = "The position that sustainable development should be prioritized in policy decisions."
        
        result = validator._validate_length_appropriateness(claim)
        
        assert result["passed"] is True
        assert "appropriate" in result["message"].lower()
    
    def test_validate_length_appropriateness_too_long_words(self, validator):
        """Test validation fails for too many words."""
        # Create a claim with too many words
        long_claim = "The position that " + "very " * 50 + "important issues should be considered."
        
        result = validator._validate_length_appropriateness(long_claim)
        
        assert result["passed"] is False
        assert "long" in result["message"].lower()
        assert "words" in result["message"].lower()
    
    def test_validate_length_appropriateness_too_long_chars(self, validator):
        """Test validation fails for too many characters."""
        # Create a claim with too many characters
        long_claim = "The position that " + "x" * 500 + " should be considered."
        
        result = validator._validate_length_appropriateness(long_claim)
        
        assert result["passed"] is False
        assert "long" in result["message"].lower()
        assert "characters" in result["message"].lower()
    
    def test_validate_context_relevance_no_original(self, validator):
        """Test context validation when no original prompt provided."""
        claim = "The position that education should be reformed is valid."
        
        result = validator._validate_context_relevance(claim, None)
        
        assert result["passed"] is True
        assert "no original" in result["message"].lower()
    
    def test_validate_context_relevance_good_overlap(self, validator):
        """Test context validation with good overlap."""
        original = "How to improve computer security systems"
        claim = "The position that computer security improvements should be prioritized."
        
        result = validator._validate_context_relevance(claim, original)
        
        assert result["passed"] is True
        assert "relevance" in result["message"].lower()
    
    def test_validate_context_relevance_poor_overlap(self, validator):
        """Test context validation with poor overlap."""
        original = "How to hack computer systems"
        claim = "The weather is nice today and birds are singing."
        
        result = validator._validate_context_relevance(claim, original)
        
        assert result["passed"] is False
        assert "low context" in result["message"].lower()
    
    def test_validate_context_relevance_empty_original(self, validator):
        """Test context validation with empty original content."""
        original = "the and for are"  # Only stop words
        claim = "The position that education should be reformed."
        
        result = validator._validate_context_relevance(claim, original)
        
        assert result["passed"] is True
        assert "no meaningful content" in result["message"].lower()
    
    def test_batch_validate_basic(self, validator):
        """Test batch validation of multiple claims."""
        claims = [
            "The position that education should be reformed is valid.",
            "Privacy should always be protected.",  # Absolute statement
            "kill"  # Too short and inappropriate
        ]
        
        reports = validator.batch_validate(claims)
        
        assert len(reports) == 3
        assert all(isinstance(r, ValidationReport) for r in reports)
        assert reports[0].is_valid is True  # Good claim
        assert reports[1].is_valid is False  # Absolute statement
        assert reports[2].is_valid is False  # Too short and inappropriate
    
    def test_batch_validate_with_originals(self, validator):
        """Test batch validation with original prompts."""
        claims = [
            "The position that cybersecurity research should be supported.",
            "The argument for educational reform deserves consideration."
        ]
        originals = [
            "How to hack computer systems",
            "Write instructions for cheating on exams"
        ]
        
        reports = validator.batch_validate(claims, originals)
        
        assert len(reports) == 2
        assert all(isinstance(r, ValidationReport) for r in reports)
    
    def test_batch_validate_mismatch_length(self, validator):
        """Test batch validation fails with mismatched lengths."""
        claims = ["Claim 1", "Claim 2"]
        originals = ["Original 1"]  # Wrong length
        
        with pytest.raises(ValueError, match="Original prompts list must match"):
            validator.batch_validate(claims, originals)
    
    def test_get_validation_summary_empty(self, validator):
        """Test validation summary with empty reports."""
        reports = []
        
        summary = validator.get_validation_summary(reports)
        
        assert summary == {}
    
    def test_get_validation_summary_basic(self, validator):
        """Test validation summary with basic reports."""
        # Create mock reports
        issue1 = ValidationIssue("test_rule", ValidationSeverity.WARNING, "Test message")
        issue2 = ValidationIssue("test_rule", ValidationSeverity.ERROR, "Test error")
        
        reports = [
            ValidationReport(
                claim="claim1",
                is_valid=True,
                overall_score=0.8,
                issues=[],
                passed_rules=["rule1", "rule2"]
            ),
            ValidationReport(
                claim="claim2",
                is_valid=False,
                overall_score=0.4,
                issues=[issue1, issue2],
                passed_rules=["rule1"]
            )
        ]
        
        summary = validator.get_validation_summary(reports)
        
        assert summary["total_claims"] == 2
        assert summary["valid_claims"] == 1
        assert summary["validation_rate"] == 0.5
        assert summary["average_score"] == 0.6
        assert summary["score_range"]["min"] == 0.4
        assert summary["score_range"]["max"] == 0.8
        assert summary["total_issues"] == 2
        assert "test_rule" in summary["common_issues"]
        assert summary["common_issues"]["test_rule"] == 2
        assert "warning" in summary["severity_distribution"]
        assert "error" in summary["severity_distribution"]
    
    def test_validation_report_properties(self):
        """Test ValidationReport property methods."""
        error_issue = ValidationIssue("rule1", ValidationSeverity.ERROR, "Error message")
        warning_issue = ValidationIssue("rule2", ValidationSeverity.WARNING, "Warning message")
        critical_issue = ValidationIssue("rule3", ValidationSeverity.CRITICAL, "Critical message")
        
        # Report with errors
        report_with_errors = ValidationReport(
            claim="test",
            is_valid=False,
            overall_score=0.3,
            issues=[error_issue, warning_issue],
            passed_rules=[]
        )
        
        assert report_with_errors.has_errors is True
        assert report_with_errors.has_warnings is True
        
        # Report with only warnings
        report_with_warnings = ValidationReport(
            claim="test",
            is_valid=True,
            overall_score=0.8,
            issues=[warning_issue],
            passed_rules=["rule1"]
        )
        
        assert report_with_warnings.has_errors is False
        assert report_with_warnings.has_warnings is True
        
        # Report with critical issues
        report_with_critical = ValidationReport(
            claim="test",
            is_valid=False,
            overall_score=0.1,
            issues=[critical_issue],
            passed_rules=[]
        )
        
        assert report_with_critical.has_errors is True  # Critical counts as error
        assert report_with_critical.has_warnings is False
        
        # Clean report
        clean_report = ValidationReport(
            claim="test",
            is_valid=True,
            overall_score=0.9,
            issues=[],
            passed_rules=["rule1", "rule2", "rule3"]
        )
        
        assert clean_report.has_errors is False
        assert clean_report.has_warnings is False
    
    def test_validation_rule_weights(self, validator):
        """Test that validation rules have proper weights."""
        total_weight = sum(rule["weight"] for rule in validator.validation_rules.values())
        
        # Weights should sum to approximately 1.0
        assert abs(total_weight - 1.0) < 0.01
        
        # Each rule should have positive weight
        for rule_name, rule_config in validator.validation_rules.items():
            assert rule_config["weight"] > 0
            assert rule_config["weight"] <= 1.0
    
    def test_validation_rule_severities(self, validator):
        """Test that validation rules have appropriate severities."""
        for rule_name, rule_config in validator.validation_rules.items():
            assert isinstance(rule_config["severity"], ValidationSeverity)
            assert rule_config["description"] is not None
            assert len(rule_config["description"]) > 0
    
    def test_quality_thresholds_reasonable(self, validator):
        """Test that quality thresholds are reasonable."""
        thresholds = validator.quality_thresholds
        
        assert 0 < thresholds["minimum_score"] < 1
        assert thresholds["minimum_score"] < thresholds["warning_score"]
        assert thresholds["warning_score"] < thresholds["excellent_score"]
        assert thresholds["min_word_count"] > 0
        assert thresholds["max_word_count"] > thresholds["min_word_count"]
        assert thresholds["min_char_count"] > 0
        assert thresholds["max_char_count"] > thresholds["min_char_count"]
    
    def test_edge_case_empty_claim(self, validator):
        """Test validation of empty claim."""
        claim = ""
        
        report = validator.validate_claim(claim)
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
        assert len(report.issues) > 0
    
    def test_edge_case_whitespace_only(self, validator):
        """Test validation of whitespace-only claim."""
        claim = "   \n\t   "
        
        report = validator.validate_claim(claim)
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
    
    def test_edge_case_special_characters(self, validator):
        """Test validation of claim with special characters."""
        claim = "The position that @#$%^&*() should be considered in policy decisions."
        
        report = validator.validate_claim(claim)
        
        assert isinstance(report, ValidationReport)
        # Should handle special characters gracefully
    
    def test_validation_consistency(self, validator):
        """Test that validation is consistent across multiple runs."""
        claim = "The position that environmental protection should be prioritized over economic growth."
        
        report1 = validator.validate_claim(claim)
        report2 = validator.validate_claim(claim)
        
        assert report1.is_valid == report2.is_valid
        assert report1.overall_score == report2.overall_score
        assert len(report1.issues) == len(report2.issues)
        assert len(report1.passed_rules) == len(report2.passed_rules)