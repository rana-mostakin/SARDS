# Contributing to SARDS

Thank you for your interest in contributing. SARDS is an open-source
satellite analytics project and welcomes contributions of all kinds.

---

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)

---

## Code of Conduct

Be respectful, constructive, and inclusive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/).

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SARDS.git
   cd SARDS
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/rana-mostakin/SARDS.git
   ```

---

## Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate           # Windows

# Install all dependencies including dev tools
pip install -r requirements.txt
pip install pytest pytest-cov flake8

# Generate demo data
python scripts/generate_demo_data.py
```

---

## Making Changes

1. Create a **feature branch** from `develop`:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, commented code.

3. Add or update tests in `tests/` for new functionality.

4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```

5. Run the full pipeline to verify nothing is broken:
   ```bash
   python main.py --quiet
   ```

6. Commit with a descriptive message:
   ```bash
   git commit -m "feat: add Sentinel-2 band compositor module"
   ```

---

## Testing

- All new modules **must** have corresponding unit tests in `tests/`
- Tests must pass before a PR will be merged
- Aim for > 80% coverage on new code

```bash
# Run tests with coverage report
pytest tests/ --cov=src --cov-report=term-missing -v
```

---

## Pull Request Process

1. Push your branch: `git push origin feature/your-feature-name`
2. Open a Pull Request against the `develop` branch (not `main`)
3. Fill in the PR template completely
4. Ensure all CI checks pass (lint + tests + pipeline smoke test)
5. Request a review from a maintainer
6. Address review feedback promptly

PRs are merged into `develop` and then periodically released to `main`.

---

## Code Style

- Follow **PEP 8** with max line length of **100 characters**
- Use **type hints** for all public function signatures
- Write **docstrings** for all public classes and methods
- Prefer **explicit** over implicit; **readable** over clever
- Group imports: stdlib -> third-party -> local (with blank lines between)

Example:
```python
def detect(
    self,
    before: SatelliteImage,
    after : SatelliteImage,
    align : bool = True
) -> ChangeResult:
    """
    Run change detection between two satellite images.

    Parameters
    ----------
    before : Earlier image.
    after  : Later image.
    align  : Apply ECC co-registration before comparison.

    Returns
    -------
    ChangeResult
    """
    ...
```

---

## Good First Issues

Look for issues labelled [`good first issue`](../../issues?q=label%3A"good+first+issue")
for beginner-friendly contributions.

---

Thank you for helping make SARDS better.