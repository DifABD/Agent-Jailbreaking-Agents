"""
JailbreakBench data loading utilities.

This module provides utilities for loading and preprocessing JailbreakBench
dataset for use in the agent jailbreaking research system.
"""

import logging
import json
import csv
import pickle
import random
from typing import List, Dict, Optional, Iterator, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetFormat(Enum):
    """Supported dataset formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"


@dataclass
class JailbreakBenchEntry:
    """Represents a single entry from the JailbreakBench dataset."""
    
    # Core fields
    prompt: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    
    # Metadata fields
    source: Optional[str] = None
    difficulty: Optional[str] = None
    tags: Optional[List[str]] = None
    
    # Processing fields
    entry_id: Optional[str] = None
    is_valid: bool = True
    validation_notes: Optional[List[str]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.entry_id is None:
            # Generate unique ID based on prompt content
            self.entry_id = self._generate_id()
        
        if self.tags is None:
            self.tags = []
        
        if self.validation_notes is None:
            self.validation_notes = []
    
    def _generate_id(self) -> str:
        """Generate a unique ID for this entry."""
        content = f"{self.prompt}_{self.category}_{self.subcategory}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class DatasetStats:
    """Statistics about a loaded dataset."""
    
    total_entries: int
    valid_entries: int
    invalid_entries: int
    categories: Dict[str, int]
    subcategories: Dict[str, int]
    sources: Dict[str, int]
    average_prompt_length: float
    validation_issues: Dict[str, int]
    length_distribution: Dict[str, int]
    quality_score: float
    processing_time: float


class JailbreakBenchLoader:
    """
    Loads and preprocesses JailbreakBench dataset.
    
    This class handles loading data from various formats, validating entries,
    and preprocessing for use in the agent jailbreaking research system.
    """
    
    def __init__(self, data_path: Optional[Path] = None, default_format: DatasetFormat = DatasetFormat.JSON):
        """
        Initialize the JailbreakBench loader.
        
        Args:
            data_path: Optional path to the dataset directory or file
            default_format: Default format to assume for data files
        """
        self.data_path = Path(data_path) if data_path else None
        self.default_format = default_format
        self.validation_rules = self._initialize_validation_rules()
        
        logger.info(f"JailbreakBenchLoader initialized with path: {self.data_path}")
    
    def _initialize_validation_rules(self) -> Dict[str, callable]:
        """Initialize validation rules for dataset entries."""
        return {
            "non_empty_prompt": self._validate_non_empty_prompt,
            "reasonable_length": self._validate_reasonable_length,
            "valid_encoding": self._validate_valid_encoding,
            "no_malformed_content": self._validate_no_malformed_content
        }
    
    def load_dataset(self, file_path: Optional[Path] = None, format_type: Optional[DatasetFormat] = None) -> List[JailbreakBenchEntry]:
        """
        Load the JailbreakBench dataset from file.
        
        Args:
            file_path: Optional specific file path to load
            format_type: Optional format type override
            
        Returns:
            List of JailbreakBenchEntry objects
        """
        # Determine file path
        if file_path:
            target_path = Path(file_path)
        elif self.data_path:
            target_path = self.data_path
        else:
            raise ValueError("No data path specified")
        
        # Determine format
        if format_type:
            file_format = format_type
        else:
            file_format = self._detect_format(target_path)
        
        logger.info(f"Loading dataset from {target_path} in {file_format.value} format")
        
        # Load based on format
        if file_format == DatasetFormat.JSON:
            entries = self._load_json(target_path)
        elif file_format == DatasetFormat.JSONL:
            entries = self._load_jsonl(target_path)
        elif file_format == DatasetFormat.CSV:
            entries = self._load_csv(target_path)
        elif file_format == DatasetFormat.TSV:
            entries = self._load_tsv(target_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Loaded {len(entries)} entries from dataset")
        return entries
    
    def _detect_format(self, file_path: Path) -> DatasetFormat:
        """Detect file format based on extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return DatasetFormat.JSON
        elif suffix == '.jsonl':
            return DatasetFormat.JSONL
        elif suffix == '.csv':
            return DatasetFormat.CSV
        elif suffix == '.tsv':
            return DatasetFormat.TSV
        else:
            logger.warning(f"Unknown file extension {suffix}, using default format {self.default_format.value}")
            return self.default_format
    
    def _load_json(self, file_path: Path) -> List[JailbreakBenchEntry]:
        """Load dataset from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = []
            if isinstance(data, list):
                # Array of entries
                for item in data:
                    entry = self._parse_entry(item)
                    if entry:
                        entries.append(entry)
            elif isinstance(data, dict):
                # Single entry or nested structure
                if 'data' in data:
                    # Nested structure with data field
                    for item in data['data']:
                        entry = self._parse_entry(item)
                        if entry:
                            entries.append(entry)
                else:
                    # Single entry
                    entry = self._parse_entry(data)
                    if entry:
                        entries.append(entry)
            
            return entries
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise
    
    def _load_jsonl(self, file_path: Path) -> List[JailbreakBenchEntry]:
        """Load dataset from JSONL file."""
        entries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        entry = self._parse_entry(data)
                        if entry:
                            entries.append(entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                        continue
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to load JSONL file {file_path}: {e}")
            raise
    
    def _load_csv(self, file_path: Path) -> List[JailbreakBenchEntry]:
        """Load dataset from CSV file."""
        return self._load_delimited(file_path, delimiter=',')
    
    def _load_tsv(self, file_path: Path) -> List[JailbreakBenchEntry]:
        """Load dataset from TSV file."""
        return self._load_delimited(file_path, delimiter='\t')
    
    def _load_delimited(self, file_path: Path, delimiter: str) -> List[JailbreakBenchEntry]:
        """Load dataset from delimited file (CSV/TSV)."""
        entries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        entry = self._parse_entry(row)
                        if entry:
                            entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Failed to parse row {row_num} in {file_path}: {e}")
                        continue
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to load delimited file {file_path}: {e}")
            raise
    
    def _parse_entry(self, data: Dict[str, Any]) -> Optional[JailbreakBenchEntry]:
        """Parse a single entry from raw data."""
        try:
            # Extract prompt (try various field names)
            prompt = None
            for field in ['prompt', 'text', 'content', 'query', 'instruction']:
                if field in data and data[field]:
                    prompt = str(data[field]).strip()
                    break
            
            if not prompt:
                logger.warning("Entry missing prompt field")
                return None
            
            # Extract other fields
            category = data.get('category') or data.get('type')
            subcategory = data.get('subcategory') or data.get('subtype')
            source = data.get('source') or data.get('dataset')
            difficulty = data.get('difficulty') or data.get('level')
            
            # Handle tags
            tags = data.get('tags', [])
            if isinstance(tags, str):
                # Parse comma-separated tags
                tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
            elif not isinstance(tags, list):
                tags = []
            
            # Create entry
            entry = JailbreakBenchEntry(
                prompt=prompt,
                category=category,
                subcategory=subcategory,
                source=source,
                difficulty=difficulty,
                tags=tags
            )
            
            return entry
            
        except Exception as e:
            logger.warning(f"Failed to parse entry: {e}")
            return None
    
    def preprocess_data(self, data: List[JailbreakBenchEntry]) -> List[JailbreakBenchEntry]:
        """
        Preprocess the loaded dataset entries.
        
        Args:
            data: Raw dataset entries
            
        Returns:
            Preprocessed dataset entries
        """
        logger.info(f"Preprocessing {len(data)} dataset entries")
        
        processed_entries = []
        
        for entry in data:
            # Validate entry
            validation_result = self._validate_entry(entry)
            entry.is_valid = validation_result["is_valid"]
            entry.validation_notes = validation_result["notes"]
            
            # Clean and normalize prompt
            entry.prompt = self._clean_prompt(entry.prompt)
            
            # Normalize categories
            if entry.category:
                entry.category = self._normalize_category(entry.category)
            if entry.subcategory:
                entry.subcategory = self._normalize_category(entry.subcategory)
            
            # Add processing tags
            entry.tags.extend(self._generate_processing_tags(entry))
            
            processed_entries.append(entry)
        
        logger.info(f"Preprocessing complete. {len(processed_entries)} entries processed")
        return processed_entries
    
    def _validate_entry(self, entry: JailbreakBenchEntry) -> Dict[str, Any]:
        """Validate a dataset entry against quality rules."""
        validation_notes = []
        is_valid = True
        
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(entry)
                if not result["passed"]:
                    is_valid = False
                    validation_notes.append(f"{rule_name}: {result['message']}")
                else:
                    validation_notes.append(f"{rule_name}: passed")
            except Exception as e:
                logger.warning(f"Validation rule {rule_name} failed for entry {entry.entry_id}: {e}")
                validation_notes.append(f"{rule_name}: validation error")
        
        return {
            "is_valid": is_valid,
            "notes": validation_notes
        }
    
    def _validate_non_empty_prompt(self, entry: JailbreakBenchEntry) -> Dict[str, Any]:
        """Validate that prompt is not empty."""
        if not entry.prompt or not entry.prompt.strip():
            return {
                "passed": False,
                "message": "Prompt is empty or whitespace only"
            }
        
        return {
            "passed": True,
            "message": "Prompt is non-empty"
        }
    
    def _validate_reasonable_length(self, entry: JailbreakBenchEntry) -> Dict[str, Any]:
        """Validate that prompt has reasonable length."""
        prompt_length = len(entry.prompt)
        
        if prompt_length < 10:
            return {
                "passed": False,
                "message": f"Prompt too short ({prompt_length} characters)"
            }
        
        if prompt_length > 5000:
            return {
                "passed": False,
                "message": f"Prompt too long ({prompt_length} characters)"
            }
        
        return {
            "passed": True,
            "message": f"Prompt length appropriate ({prompt_length} characters)"
        }
    
    def _validate_valid_encoding(self, entry: JailbreakBenchEntry) -> Dict[str, Any]:
        """Validate that prompt has valid text encoding."""
        try:
            # Try to encode/decode to check for encoding issues
            entry.prompt.encode('utf-8').decode('utf-8')
            
            # Check for common encoding artifacts
            artifacts = ['�', '\ufffd', '\x00']
            for artifact in artifacts:
                if artifact in entry.prompt:
                    return {
                        "passed": False,
                        "message": f"Contains encoding artifact: {repr(artifact)}"
                    }
            
            return {
                "passed": True,
                "message": "Valid text encoding"
            }
            
        except UnicodeError as e:
            return {
                "passed": False,
                "message": f"Invalid text encoding: {e}"
            }
    
    def _validate_no_malformed_content(self, entry: JailbreakBenchEntry) -> Dict[str, Any]:
        """Validate that prompt doesn't contain malformed content."""
        # Check for excessive whitespace
        if re.search(r'\s{10,}', entry.prompt):
            return {
                "passed": False,
                "message": "Contains excessive whitespace"
            }
        
        # Check for repeated characters (potential corruption)
        if re.search(r'(.)\1{20,}', entry.prompt):
            return {
                "passed": False,
                "message": "Contains excessive character repetition"
            }
        
        # Check for HTML/XML tags (might indicate parsing issues)
        if re.search(r'<[^>]+>', entry.prompt):
            return {
                "passed": False,
                "message": "Contains HTML/XML tags"
            }
        
        return {
            "passed": True,
            "message": "No malformed content detected"
        }
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize prompt text."""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', prompt)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Remove common artifacts
        cleaned = cleaned.replace('\ufffd', '')  # Unicode replacement character
        cleaned = cleaned.replace('\x00', '')   # Null bytes
        
        return cleaned
    
    def _normalize_category(self, category: str) -> str:
        """Normalize category names."""
        # Convert to lowercase and replace separators
        normalized = category.lower().replace('_', ' ').replace('-', ' ')
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _generate_processing_tags(self, entry: JailbreakBenchEntry) -> List[str]:
        """Generate additional tags based on prompt content."""
        tags = []
        
        # Length-based tags
        prompt_length = len(entry.prompt)
        if prompt_length < 50:
            tags.append("short_prompt")
        elif prompt_length > 500:
            tags.append("long_prompt")
        
        # Content-based tags
        prompt_lower = entry.prompt.lower()
        
        if any(word in prompt_lower for word in ['write', 'create', 'generate', 'make']):
            tags.append("instruction_request")
        
        if any(word in prompt_lower for word in ['how to', 'step by step', 'guide', 'tutorial']):
            tags.append("how_to_request")
        
        if any(word in prompt_lower for word in ['illegal', 'harmful', 'dangerous', 'unethical']):
            tags.append("explicit_harm")
        
        return tags
    
    def batch_process(self, file_paths: List[Path], batch_size: int = 1000) -> Iterator[List[JailbreakBenchEntry]]:
        """
        Process multiple files in batches.
        
        Args:
            file_paths: List of file paths to process
            batch_size: Number of entries per batch
            
        Yields:
            Batches of processed entries
        """
        current_batch = []
        
        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            
            try:
                entries = self.load_dataset(file_path)
                processed_entries = self.preprocess_data(entries)
                
                for entry in processed_entries:
                    current_batch.append(entry)
                    
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []
                        
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue
        
        # Yield remaining entries
        if current_batch:
            yield current_batch
    
    def get_dataset_stats(self, entries: List[JailbreakBenchEntry]) -> DatasetStats:
        """
        Generate statistics about the dataset.
        
        Args:
            entries: List of dataset entries
            
        Returns:
            DatasetStats object with comprehensive statistics
        """
        if not entries:
            return DatasetStats(
                total_entries=0,
                valid_entries=0,
                invalid_entries=0,
                categories={},
                subcategories={},
                sources={},
                average_prompt_length=0.0,
                validation_issues={}
            )
        
        # Basic counts
        total_entries = len(entries)
        valid_entries = sum(1 for e in entries if e.is_valid)
        invalid_entries = total_entries - valid_entries
        
        # Category distributions
        categories = {}
        subcategories = {}
        sources = {}
        
        for entry in entries:
            if entry.category:
                categories[entry.category] = categories.get(entry.category, 0) + 1
            if entry.subcategory:
                subcategories[entry.subcategory] = subcategories.get(entry.subcategory, 0) + 1
            if entry.source:
                sources[entry.source] = sources.get(entry.source, 0) + 1
        
        # Length statistics
        prompt_lengths = [len(entry.prompt) for entry in entries]
        average_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
        
        # Length distribution
        length_distribution = {
            "short (<50 chars)": sum(1 for l in prompt_lengths if l < 50),
            "medium (50-200 chars)": sum(1 for l in prompt_lengths if 50 <= l < 200),
            "long (200-500 chars)": sum(1 for l in prompt_lengths if 200 <= l < 500),
            "very_long (>=500 chars)": sum(1 for l in prompt_lengths if l >= 500)
        }
        
        # Validation issues
        validation_issues = {}
        for entry in entries:
            if not entry.is_valid and entry.validation_notes:
                for note in entry.validation_notes:
                    if ":" in note and not note.endswith("passed"):
                        issue_type = note.split(":")[0]
                        validation_issues[issue_type] = validation_issues.get(issue_type, 0) + 1
        
        # Quality metrics
        quality_metrics = self.get_quality_metrics(entries)
        quality_score = quality_metrics.get('overall_quality_score', 0.0)
        
        return DatasetStats(
            total_entries=total_entries,
            valid_entries=valid_entries,
            invalid_entries=invalid_entries,
            categories=categories,
            subcategories=subcategories,
            sources=sources,
            average_prompt_length=average_prompt_length,
            validation_issues=validation_issues,
            length_distribution=length_distribution,
            quality_score=quality_score,
            processing_time=0.0  # Will be set by caller if needed
        )
    
    def export_processed_data(self, entries: List[JailbreakBenchEntry], output_path: Path, format_type: DatasetFormat = DatasetFormat.JSON) -> None:
        """
        Export processed data to file.
        
        Args:
            entries: Processed dataset entries
            output_path: Output file path
            format_type: Output format
        """
        logger.info(f"Exporting {len(entries)} entries to {output_path} in {format_type.value} format")
        
        # Convert entries to dictionaries
        data = [entry.to_dict() for entry in entries]
        
        if format_type == DatasetFormat.JSON:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format_type == DatasetFormat.JSONL:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry_data in data:
                    json.dump(entry_data, f, ensure_ascii=False)
                    f.write('\n')
        
        elif format_type in [DatasetFormat.CSV, DatasetFormat.TSV]:
            delimiter = ',' if format_type == DatasetFormat.CSV else '\t'
            
            if data:
                fieldnames = data[0].keys()
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                    writer.writeheader()
                    writer.writerows(data)
        
        logger.info(f"Export complete: {output_path}")
    
    def filter_entries(self, entries: List[JailbreakBenchEntry], **filters) -> List[JailbreakBenchEntry]:
        """
        Filter entries based on criteria.
        
        Args:
            entries: List of entries to filter
            **filters: Filter criteria (valid_only, category, source, etc.)
            
        Returns:
            Filtered list of entries
        """
        filtered = entries
        
        # Filter by validity
        if filters.get('valid_only', False):
            filtered = [e for e in filtered if e.is_valid]
        
        # Filter by category
        if 'category' in filters:
            category = filters['category']
            filtered = [e for e in filtered if e.category == category]
        
        # Filter by source
        if 'source' in filters:
            source = filters['source']
            filtered = [e for e in filtered if e.source == source]
        
        # Filter by tags
        if 'has_tag' in filters:
            tag = filters['has_tag']
            filtered = [e for e in filtered if tag in e.tags]
        
        # Filter by prompt length
        if 'min_length' in filters:
            min_len = filters['min_length']
            filtered = [e for e in filtered if len(e.prompt) >= min_len]
        
        if 'max_length' in filters:
            max_len = filters['max_length']
            filtered = [e for e in filtered if len(e.prompt) <= max_len]
        
        logger.info(f"Filtered {len(entries)} entries to {len(filtered)} entries")
        return filtered
    
    def sample_data(self, entries: List[JailbreakBenchEntry], 
                   sample_size: Optional[int] = None, 
                   sample_ratio: Optional[float] = None,
                   stratify_by: Optional[str] = None,
                   random_seed: Optional[int] = None) -> List[JailbreakBenchEntry]:
        """
        Sample data from the dataset.
        
        Args:
            entries: List of entries to sample from
            sample_size: Absolute number of samples to take
            sample_ratio: Ratio of samples to take (0.0 to 1.0)
            stratify_by: Field to stratify sampling by ('category', 'source', etc.)
            random_seed: Random seed for reproducible sampling
            
        Returns:
            Sampled list of entries
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        if not entries:
            return []
        
        # Determine sample size
        if sample_size is not None:
            target_size = min(sample_size, len(entries))
        elif sample_ratio is not None:
            target_size = int(len(entries) * sample_ratio)
        else:
            raise ValueError("Must specify either sample_size or sample_ratio")
        
        if stratify_by:
            # Stratified sampling
            return self._stratified_sample(entries, target_size, stratify_by)
        else:
            # Simple random sampling
            return random.sample(entries, target_size)
    
    def _stratified_sample(self, entries: List[JailbreakBenchEntry], 
                          target_size: int, stratify_field: str) -> List[JailbreakBenchEntry]:
        """Perform stratified sampling based on a field."""
        # Group entries by stratification field
        groups = {}
        for entry in entries:
            key = getattr(entry, stratify_field, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)
        
        # Calculate samples per group
        sampled_entries = []
        remaining_size = target_size
        
        for group_key, group_entries in groups.items():
            if remaining_size <= 0:
                break
            
            # Proportional allocation
            group_ratio = len(group_entries) / len(entries)
            group_sample_size = max(1, int(target_size * group_ratio))
            group_sample_size = min(group_sample_size, len(group_entries), remaining_size)
            
            group_sample = random.sample(group_entries, group_sample_size)
            sampled_entries.extend(group_sample)
            remaining_size -= group_sample_size
        
        return sampled_entries
    
    def split_data(self, entries: List[JailbreakBenchEntry], 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, 
                   test_ratio: float = 0.15,
                   stratify_by: Optional[str] = None,
                   random_seed: Optional[int] = None) -> Tuple[List[JailbreakBenchEntry], 
                                                              List[JailbreakBenchEntry], 
                                                              List[JailbreakBenchEntry]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            entries: List of entries to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            stratify_by: Field to stratify split by
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train, validation, test) entry lists
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if random_seed is not None:
            random.seed(random_seed)
        
        if stratify_by:
            return self._stratified_split(entries, train_ratio, val_ratio, test_ratio, stratify_by)
        else:
            # Simple random split
            shuffled = entries.copy()
            random.shuffle(shuffled)
            
            n_total = len(shuffled)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_set = shuffled[:n_train]
            val_set = shuffled[n_train:n_train + n_val]
            test_set = shuffled[n_train + n_val:]
            
            return train_set, val_set, test_set
    
    def _stratified_split(self, entries: List[JailbreakBenchEntry], 
                         train_ratio: float, val_ratio: float, test_ratio: float,
                         stratify_field: str) -> Tuple[List[JailbreakBenchEntry], 
                                                      List[JailbreakBenchEntry], 
                                                      List[JailbreakBenchEntry]]:
        """Perform stratified split based on a field."""
        # Group entries by stratification field
        groups = {}
        for entry in entries:
            key = getattr(entry, stratify_field, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)
        
        train_set, val_set, test_set = [], [], []
        
        for group_entries in groups.values():
            # Shuffle group
            shuffled_group = group_entries.copy()
            random.shuffle(shuffled_group)
            
            # Split group
            n_group = len(shuffled_group)
            n_train = int(n_group * train_ratio)
            n_val = int(n_group * val_ratio)
            
            train_set.extend(shuffled_group[:n_train])
            val_set.extend(shuffled_group[n_train:n_train + n_val])
            test_set.extend(shuffled_group[n_train + n_val:])
        
        return train_set, val_set, test_set
    
    def cache_processed_data(self, entries: List[JailbreakBenchEntry], 
                           cache_path: Path, 
                           metadata: Optional[Dict] = None) -> None:
        """
        Cache processed data to disk for faster loading.
        
        Args:
            entries: Processed entries to cache
            cache_path: Path to save cache file
            metadata: Optional metadata to include with cache
        """
        cache_data = {
            'entries': entries,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Cached {len(entries)} entries to {cache_path}")
    
    def load_cached_data(self, cache_path: Path) -> Optional[List[JailbreakBenchEntry]]:
        """
        Load cached processed data from disk.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            List of cached entries or None if cache invalid/missing
        """
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache structure
            if not isinstance(cache_data, dict) or 'entries' not in cache_data:
                logger.warning(f"Invalid cache structure in {cache_path}")
                return None
            
            entries = cache_data['entries']
            logger.info(f"Loaded {len(entries)} entries from cache {cache_path}")
            return entries
            
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def get_quality_metrics(self, entries: List[JailbreakBenchEntry]) -> Dict[str, float]:
        """
        Calculate quality metrics for the dataset.
        
        Args:
            entries: List of entries to analyze
            
        Returns:
            Dictionary of quality metrics
        """
        if not entries:
            return {}
        
        # Basic quality metrics
        valid_ratio = sum(1 for e in entries if e.is_valid) / len(entries)
        
        # Length distribution metrics
        lengths = [len(e.prompt) for e in entries]
        avg_length = sum(lengths) / len(lengths)
        length_std = (sum((l - avg_length) ** 2 for l in lengths) / len(lengths)) ** 0.5
        
        # Category distribution metrics
        categories = [e.category for e in entries if e.category]
        category_diversity = len(set(categories)) / len(categories) if categories else 0
        
        # Validation issue metrics
        total_issues = sum(len(e.validation_notes) for e in entries if e.validation_notes)
        avg_issues_per_entry = total_issues / len(entries)
        
        # Content quality metrics
        unique_prompts = len(set(e.prompt for e in entries))
        uniqueness_ratio = unique_prompts / len(entries)
        
        return {
            'validity_ratio': valid_ratio,
            'average_length': avg_length,
            'length_std_dev': length_std,
            'category_diversity': category_diversity,
            'avg_issues_per_entry': avg_issues_per_entry,
            'uniqueness_ratio': uniqueness_ratio,
            'overall_quality_score': (valid_ratio + category_diversity + uniqueness_ratio) / 3
        }
    
    def process_dataset_pipeline(self, 
                               file_paths: List[Path],
                               output_dir: Optional[Path] = None,
                               cache_dir: Optional[Path] = None,
                               validation_config: Optional[Dict] = None,
                               sample_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete data processing pipeline for JailbreakBench dataset.
        
        Args:
            file_paths: List of dataset files to process
            output_dir: Directory to save processed data
            cache_dir: Directory for caching intermediate results
            validation_config: Configuration for validation rules
            sample_config: Configuration for data sampling
            
        Returns:
            Dictionary containing processing results and statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting dataset processing pipeline with {len(file_paths)} files")
        
        # Initialize results
        all_entries = []
        processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_entries_loaded': 0,
            'total_entries_valid': 0,
            'processing_errors': []
        }
        
        # Process each file
        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")
                
                # Check cache first
                cache_path = None
                if cache_dir:
                    cache_path = cache_dir / f"{file_path.stem}_processed.pkl"
                    cached_entries = self.load_cached_data(cache_path)
                    if cached_entries:
                        all_entries.extend(cached_entries)
                        processing_stats['files_processed'] += 1
                        processing_stats['total_entries_loaded'] += len(cached_entries)
                        continue
                
                # Load and process data
                raw_entries = self.load_dataset(file_path)
                processed_entries = self.preprocess_data(raw_entries)
                
                # Apply custom validation if configured
                if validation_config:
                    processed_entries = self._apply_custom_validation(processed_entries, validation_config)
                
                # Cache processed data
                if cache_path:
                    self.cache_processed_data(processed_entries, cache_path, {
                        'source_file': str(file_path),
                        'processing_time': datetime.now().isoformat()
                    })
                
                all_entries.extend(processed_entries)
                processing_stats['files_processed'] += 1
                processing_stats['total_entries_loaded'] += len(processed_entries)
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                processing_stats['files_failed'] += 1
                processing_stats['processing_errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        # Apply sampling if configured
        if sample_config and all_entries:
            logger.info("Applying data sampling configuration")
            all_entries = self.sample_data(all_entries, **sample_config)
        
        # Generate comprehensive statistics
        dataset_stats = self.get_dataset_stats(all_entries)
        quality_metrics = self.get_quality_metrics(all_entries)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        dataset_stats.processing_time = processing_time
        
        # Count valid entries
        processing_stats['total_entries_valid'] = sum(1 for e in all_entries if e.is_valid)
        
        # Save processed data if output directory specified
        if output_dir and all_entries:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full dataset
            full_output_path = output_dir / "processed_dataset.json"
            self.export_processed_data(all_entries, full_output_path, DatasetFormat.JSON)
            
            # Save valid entries only
            valid_entries = [e for e in all_entries if e.is_valid]
            if valid_entries:
                valid_output_path = output_dir / "valid_dataset.json"
                self.export_processed_data(valid_entries, valid_output_path, DatasetFormat.JSON)
            
            # Save statistics
            stats_path = output_dir / "dataset_statistics.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                stats_dict = {
                    'dataset_stats': asdict(dataset_stats),
                    'quality_metrics': quality_metrics,
                    'processing_stats': processing_stats
                }
                json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline complete. Processed {len(all_entries)} entries in {processing_time:.2f}s")
        
        return {
            'entries': all_entries,
            'dataset_stats': dataset_stats,
            'quality_metrics': quality_metrics,
            'processing_stats': processing_stats,
            'processing_time': processing_time
        }
    
    def _apply_custom_validation(self, entries: List[JailbreakBenchEntry], 
                               validation_config: Dict) -> List[JailbreakBenchEntry]:
        """Apply custom validation rules based on configuration."""
        # This method can be extended to apply custom validation rules
        # based on the validation_config dictionary
        
        # Example: Filter by minimum length
        if 'min_length' in validation_config:
            min_len = validation_config['min_length']
            for entry in entries:
                if len(entry.prompt) < min_len:
                    entry.is_valid = False
                    entry.validation_notes.append(f"custom_validation: prompt too short ({len(entry.prompt)} < {min_len})")
        
        # Example: Filter by category
        if 'allowed_categories' in validation_config:
            allowed = set(validation_config['allowed_categories'])
            for entry in entries:
                if entry.category and entry.category not in allowed:
                    entry.is_valid = False
                    entry.validation_notes.append(f"custom_validation: category '{entry.category}' not allowed")
        
        return entries