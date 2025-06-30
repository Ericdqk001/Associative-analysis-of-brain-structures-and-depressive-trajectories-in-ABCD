# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a neuroscience research project that performs associative analysis of brain structural phenotypes and different depressive trajectories in adolescents using ABCD (Adolescent Brain Cognitive Development) Study data.

## Development Commands

### Code Quality

- **Lint**: `ruff check src/`
- **Format**: `ruff format src/`
- **Pre-commit hooks**: `pre-commit run --all-files`
- **Always use IDE diagnostics to validate code after implementation**

### Package Management

- **Install dependencies**: `pip install -e .` (editable install)
- **Update dependencies**: `pip-compile --output-file=requirements.txt pyproject.toml`

### Testing

No specific test framework configured. Check with the user for testing approach if needed.

## Architecture and Code Structure

### Core Data Processing Pipeline

The main preprocessing logic is in `src/preprocess/scripts/prepare_data.py` which implements a comprehensive neuroimaging data preprocessing pipeline:

1. **Data Quality Control**: Applies strict inclusion criteria for T1w and dMRI data, removes subjects with neurological issues (MRI clinical report score < 3), and excludes intersex subjects
2. **Multi-modal Imaging Features**: Processes cortical thickness, cortical volume, surface area, subcortical volume, fractional anisotropy (FA), and mean diffusivity (MD)
3. **Covariate Integration**: Adds demographics, imaging device ID, age terms, family relationships, and genetic ancestry PCs
4. **Polygenic Risk Scores**: Integrates PRS data for depression risk analysis
5. **Data Harmonization**: Keeps only unrelated subjects (one per family) and standardizes continuous variables
6. **Long-form Transformation**: Creates bilateral hemisphere data for mixed-effects modeling

### Key Data Transformations

- **DTI Feature Renaming**: Uses regex parsing to create standardized feature names with hemisphere suffixes (lh/rh)
- **Feature Selection**: Separates bilateral features (for mixed-effects models) from unilateral features (for GLM)
- **Global Feature Removal**: Excludes whole-brain summary measures to focus on regional analyses

### File Organization

- `src/preprocess/main.py`: Entry point (currently empty)
- `src/preprocess/scripts/prepare_data.py`: Complete preprocessing pipeline
- Uses external data paths: `/Volumes/GenScotDepression/` for ABCD data storage

### Dependencies

Key neuroimaging libraries:

- `nilearn`: Neuroimaging analysis
- `neuroCombat`: Batch effect correction
- `scikit-learn`: Machine learning utilities
- `pandas`: Data manipulation

### Configuration

- Ruff configuration includes docstring and line length rules
- Supports both Google-style docstrings and SQL formatting
- Pre-commit hooks configured for code quality
