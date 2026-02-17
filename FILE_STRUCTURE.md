# File Structure and Descriptions

## Complete Project File Listing

### Root Level Files

| File | Purpose |
|------|----------|
| `README.md` | Main documentation with full API reference and usage guide |
| `QUICKSTART.md` | Quick start guide for beginners (5-10 minute setup) |
| `ADVANCED_USAGE.md` | Advanced usage patterns and complex scenarios |
| `PROJECT_OVERVIEW.md` | High-level project architecture and design |
| `FILE_STRUCTURE.md` | This file - complete file listing and descriptions |
| `requirements.txt` | Python package dependencies |
| `setup.py` | Package installation configuration |
| `pyproject.toml` | Modern Python project configuration (PEP 518) |
| `pytest.ini` | Pytest and coverage configuration |
| `Makefile` | Development tasks (install, test, format, lint) |
| `LICENSE` | MIT License |
| `.gitignore` | Git ignore rules |
| `getting_started.py` | Quick getting-started script |

### `/src` - Source Code

| File | Purpose |
|------|----------|
| `__init__.py` | Package initialization and exports |
| `data_quality_checker.py` | Main quality checker classes |
| `utils.py` | Utility functions for text processing |

#### `data_quality_checker.py` Contents
- `DataQualityChecker` class - Main evaluation class
  - `compute_rouge()` - Compute ROUGE metrics
  - `compute_bleu()` - Compute BLEU score
  - `compute_all_metrics()` - Compute all metrics
  - `batch_compute_metrics()` - Batch processing
  - `format_results()` - Format results for display
  - Helper methods for metric explanation

- `QualityMetricsAggregator` class - Aggregation utilities
  - `aggregate_rouge_scores()` - Aggregate ROUGE across samples
  - `aggregate_bleu_scores()` - Aggregate BLEU across samples

#### `utils.py` Contents
- `preprocess_text()` - Text preprocessing
- `tokenize_text()` - Simple tokenization
- `get_ngrams()` - N-gram extraction
- `calculate_precision_recall_f1()` - Token-level metrics
- `calculate_length_similarity()` - Length analysis
- `detect_data_quality_issues()` - Issue detection

### `/examples` - Usage Examples

| File | Focus Area |
|------|-----------|
| `example_1_basic_usage.py` | Basic ROUGE and BLEU computation |
| `example_2_batch_aggregation.py` | Batch processing and aggregation |
| `example_3_quality_detection.py` | Quality issue detection and analysis |

### `/tests` - Unit Tests

| File | Coverage |
|------|----------|
| `test_data_quality_checker.py` | Tests for main classes |
| `test_utils.py` | Tests for utility functions |

#### Test Classes

**test_data_quality_checker.py:**
- `TestDataQualityChecker` - Main class tests
- `TestQualityMetricsAggregator` - Aggregator tests
- `TestEdgeCases` - Edge case handling

**test_utils.py:**
- `TestTextProcessing` - Text preprocessing tests
- `TestTokenization` - Tokenization tests
- `TestNgrams` - N-gram extraction tests
- `TestPrecisionRecallF1` - Metric calculation tests
- `TestLengthSimilarity` - Length analysis tests
- `TestQualityIssueDetection` - Issue detection tests

## Dependencies Structure

```
evaluate
├── rouge metric
├── bleu metric
└── supporting utilities

numpy
└── array operations & aggregation

scikit-learn
└── optional statistical operations
```

## Key Concepts

### 1. Text Evaluation Pipeline
```
Raw Text → Preprocessing → Tokenization → N-gram Extraction → Metric Computation
```

### 2. Metric Hierarchy
```
ROUGE Family:
├── ROUGE-1 (unigrams)
├── ROUGE-2 (bigrams)
├── ROUGE-L (longest common subsequence)
└── ROUGE-S (skip-bigrams)

BLEU
├── 1-gram precision
├── 2-gram precision
├── 3-gram precision
├── 4-gram precision
└── Brevity penalty
```

### 3. Aggregation Flow
```
Individual Samples
        ↓
Compute Metrics
        ↓
Collect Scores
        ↓
Aggregate (Mean/Median/Min/Max)
        ↓
Dataset-Level Statistics
```

## Code Organization Principles

### Single Responsibility Principle
- `DataQualityChecker`: Metric computation only
- `QualityMetricsAggregator`: Aggregation only
- `utils`: Helper functions only

### Modularity
- Easy to import specific functions
- Each module is independently testable
- Clear dependencies between components

### Documentation
- Comprehensive docstrings with examples
- README with full API reference
- Multiple levels of getting started guides
- Advanced usage documentation

## Configuration Files Explained

### `pyproject.toml`
- Modern Python packaging standard (PEP 518)
- Build system configuration
- Tool configuration (black, isort, mypy)
- Test configuration
- Coverage settings

### `pytest.ini`
- Pytest discovery settings
- Test markers for categorization
- Coverage configuration
- Output format options

### `setup.py`
- Traditional package setup
- Dependency specifications
- Package metadata
- Entry points (if needed)

### `Makefile`
- Common development tasks
- Installation commands
- Testing commands
- Code quality checks
- Example running

## Getting Help Resources

| Resource | Purpose |
|----------|---------|
| `README.md` | Complete reference documentation |
| `QUICKSTART.md` | Quick setup and basic examples |
| `ADVANCED_USAGE.md` | Complex scenarios and patterns |
| `PROJECT_OVERVIEW.md` | Architecture and design |
| `examples/` | Working code examples |
| `tests/` | Usage patterns in tests |
| Docstrings | In-code documentation |

## Development Workflow Files

### For Testing
- `tests/` - All test files
- `pytest.ini` - Test configuration
- `Makefile` - Test commands

### For Code Quality
- `Makefile` - lint, format commands
- `.gitignore` - What to exclude
- `pyproject.toml` - Tool configurations

### For Documentation
- `README.md` - Main docs
- `QUICKSTART.md` - Quick start
- `ADVANCED_USAGE.md` - Advanced patterns
- `PROJECT_OVERVIEW.md` - Architecture

## File Size and Scope Guide

| Category | Files | Size | Purpose |
|----------|-------|------|---------|
| Core Implementation | 3 | ~700 lines | Main functionality |
| Tests | 2 | ~400 lines | Validation |
| Examples | 3 | ~400 lines | Usage demonstrations |
| Documentation | 5 | ~2000 lines | Learning and reference |
| Configuration | 5 | ~200 lines | Project setup |

## Quick Access Guide

### I want to...

**...get started quickly**
- Start: `QUICKSTART.md`
- Run: `getting_started.py`
- Example: `examples/example_1_basic_usage.py`

**...understand the architecture**
- Read: `PROJECT_OVERVIEW.md`
- Browse: `src/`
- Check: `README.md` → Architecture section

**...use advanced features**
- Read: `ADVANCED_USAGE.md`
- Run: `examples/example_2_batch_aggregation.py`
- Run: `examples/example_3_quality_detection.py`

**...contribute or extend**
- Setup: `make setup-dev`
- Code: `src/`
- Test: `make test`
- Quality: `make lint format`

**...run tests**
- All: `make test`
- Specific: `make test-specific TEST_FILE=test_name.py`
- Watch: Configure IDE or use pytest-watch

**...deploy or install**
- Simple: `pip install -r requirements.txt`
- Development: `pip install -e ".[dev]"`
- Production: `python setup.py install`

## Module Dependencies

```
src/
├── data_quality_checker.py
│   ├── imports: evaluate, warnings
│   ├── exports: DataQualityChecker, QualityMetricsAggregator
│   └── uses: numpy (optional)
│
└── utils.py
    ├── imports: re, collections
    ├── exports: 6 utility functions
    └── uses: numpy (for aggregation)
```

## External Dependencies

### Required
- `evaluate >= 0.4.0` - ROUGE and BLEU computation
- `numpy >= 1.21.0` - Numerical operations

### Optional (dev)
- `pytest >= 7.0` - Testing framework
- `pytest-cov >= 3.0` - Coverage reporting
- `black >= 22.0` - Code formatting
- `flake8 >= 4.0` - Linting
- `isort >= 5.0` - Import sorting
- `mypy >= 0.900` - Type checking

## Version Information

- **Python**: 3.8+
- **Project**: 0.1.0 (Alpha)
- **Last Updated**: February 2026

## Summary Statistics

- **Total Python Files**: 8 (3 source, 2 test, 3 example)
- **Total Lines of Code**: ~1500+
- **Total Lines of Tests**: ~400+
- **Total Lines of Docs**: ~2000+
- **Total Test Cases**: ~50+
- **Configuration Files**: 5
- **Documentation Files**: 5

---

For detailed information about any file, see the documentation index or specific README files.
