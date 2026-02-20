# Refactor load_config: Replace SimpleNamespace with Pydantic

## TL;DR

> **Quick Summary**: Replace the recursive `SimpleNamespace` conversion in `load_config` function with Pydantic V2's `model_validate` for type-safe configuration loading with integrated validation.
>
> **Deliverables**:
> - Pydantic V2 dependency added to `pyproject.toml`
> - Nested Pydantic BaseModel classes for configuration structure
> - Refactored `load_config` function using `model_validate`
> - Integrated validation logic via `field_validator`
> - Removed `_dict_to_namespace` function and `SimpleNamespace` import
> - Test suite demonstrating behavioral parity
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Dependency addition → Model definitions → load_config refactor → Tests → Cleanup

---

## Context

### Original Request
Replace the recursive SimpleNamespace conversion logic in `load_config` function with Pydantic's `model_validate`. The file is `src/graspkit/data_IO/processing_data_loader.py`.

### Interview Summary

**Key Discussions**:
- **Pydantic Version**: User confirmed using Pydantic V2 (latest with `model_validate` API)
- **Return Type**: Direct Pydantic BaseModel return (no SimpleNamespace wrapper)
- **Validation Strategy**: Integrate `_validate_config_data` logic into Pydantic using `field_validator`
- **Test Strategy**: TDD (RED-GREEN-REFACTOR approach)
- **Dynamic Attributes**: `cal_path` is added externally to config object (~15 attributes in `_setup_config_paths`)

**Research Findings**:
- Pydantic V2 `model_validate()` creates model instances from dict with automatic nested model handling
- Pydantic models support dot-notation attribute access (`config.field`), compatible with existing usage patterns
- `field_validator` decorator enables custom validation logic integration
- Pydantic has native `Path` type support

### Metis Review

**Identified Gaps** (addressed):
- **Critical Issue 1 - Dynamic Attribute Addition**: `_setup_config_paths()` adds `cal_path` attributes after `load_config` returns. **Resolution**: Use `ConfigDict(extra='allow')` for forward compatibility.
- **Critical Issue 2 - class_weight Key Conversion**: TOML string keys need conversion to int keys. **Resolution**: Use `field_validator` with `mode='before'` for key transformation.
- **Risk - Return Type Change**: From `SimpleNamespace` to Pydantic BaseModel. **Resolution**: Ensure all attribute access patterns work identically.
- **Risk - Backward Compatibility**: Code checking `isinstance(config, SimpleNamespace)`. **Resolution**: Pydantic models are not SimpleNamespace instances, but provide equivalent interface.
- **Edge Case - Optional Sections**: `rnucleus`, `server_settings`, `model_params` may be absent. **Resolution**: Use `Optional[ModelType]` fields.

**Guardrails Applied** (from Metis review):
- **MUST NOT**: Change `ml_initializer.py`, `ml_trainer.py`, or `ml_results_analyzer.py`
- **MUST NOT**: Alter expected TOML file format
- **MUST NOT**: Change validation error messages without explicit approval
- **MUST NOT**: Remove `load_config` from public exports in `__init__.py`
- **MUST NOT**: Add new fields to config structure

---

## Work Objectives

### Core Objective
Refactor the `load_config` function in `src/graspkit/data_IO/processing_data_loader.py` to use Pydantic V2's `model_validate` instead of recursive `SimpleNamespace` conversion, while maintaining full backward compatibility with existing code.

### Concrete Deliverables
- Updated `pyproject.toml` with `pydantic>=2.0` dependency
- Pydantic BaseModel classes: `Config`, `Target`, `CalSettings`, `Rnucleus`, `ServerSettings`, `ModelParams`, `MlConfig`, `CalPath`
- Refactored `load_config` function returning Pydantic model
- Removed `_dict_to_namespace` function
- Removed `SimpleNamespace` import
- Test suite verifying behavioral parity

### Definition of Done
- [ ] `load_config` returns Pydantic `Config` model (type annotation updated)
- [ ] All existing attribute access patterns work (`config.cal_settings.root_path`, `config.target.conf`, etc.)
- [ ] Dynamic `cal_path` attributes can be added and accessed
- [ ] Validation errors match current behavior (or are improved)
- [ ] Type conversions work identically (float, int, Path, dict keys)
- [ ] `ruff check .` passes (NumPy 2.0 compatible)
- [ ] All tests pass (TDD workflow completed)

### Must Have
- Pydantic V2 as a main dependency (not optional)
- All configuration fields properly typed with Python 3.13 style (`str | int`, not `Optional[str]`)
- Validation logic from `_validate_config_data` integrated as `field_validator`
- `cal_path` handled via `ConfigDict(extra='allow')` for dynamic attributes
- Return type annotation changed from `SimpleNamespace` to root Pydantic model
- `_dict_to_namespace` function deleted
- `SimpleNamespace` import removed

### Must NOT Have (Guardrails)
- **Do NOT modify**: `ml_initializer.py`, `ml_trainer.py`, `ml_results_analyzer.py`
- **Do NOT change**: TOML file format expectations
- **Do NOT alter**: Validation error message language/format without user approval
- **Do NOT remove**: `load_config` from `data_IO/__init__.py` exports
- **Do NOT add**: New configuration fields or features
- **Do NOT change**: Function signature parameter names (`config_path` must remain)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision
- **Infrastructure exists**: NO (no test file for `load_config`)
- **Automated tests**: YES (TDD) - Each task includes test cases as part of acceptance criteria.
- **Framework**: pytest (already in dev dependencies)
- **If TDD**: Each task follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Backend/Configuration**: Use Bash (python + pytest) — Run tests, assert behavior, check validation
- **Library/Module**: Use Bash (python REPL) — Import, call functions, compare output
- **Each scenario** = exact tool + exact steps + exact assertions + evidence path

---

## Execution Strategy

### Parallel Execution Waves

> Maximize throughput by grouping independent tasks into parallel waves.
> Each wave completes before the next begins.
> Target: 5-8 tasks per wave. Fewer than 3 per wave (except final) = under-splitting.

```
Wave 1 (Start Immediately — dependency + test setup):
├── Task 1: Add Pydantic dependency to pyproject.toml [quick]
├── Task 2: Create test file structure and pytest configuration [quick]
└── Task 3: Write failing tests for load_config behavior [quick]

Wave 2 (After Wave 1 — Pydantic model definitions, MAX PARALLEL):
├── Task 4: Define Target, CalPath, MlConfig Pydantic models [quick]
├── Task 5: Define CalSettings, Rnucleus, ServerSettings models [quick]
├── Task 6: Define ModelParams Pydantic model [quick]
├── Task 7: Define root Config model with all sections [quick]
└── Task 8: Create config_models.py module for all models [quick]

Wave 3 (After Wave 2 — load_config refactor):
├── Task 9: Update load_config imports and add Pydantic models [quick]
├── Task 10: Implement load_config with model_validate (keep old code) [deep]
└── Task 11: Integrate validation logic via field_validator [deep]

Wave 4 (After Wave 3 — cleanup):
├── Task 12: Remove _dict_to_namespace function [quick]
├── Task 13: Remove SimpleNamespace import [quick]
├── Task 14: Remove _process_config_data function (moved to validators) [quick]
├── Task 15: Remove _validate_config_data function (moved to validators) [quick]
└── Task 16: Update type hints to Python 3.13 style [quick]

Wave 5 (After Wave 4 — verification):
├── Task 17: Run all tests and verify behavior parity [deep]
├── Task 18: Integration test with actual TOML config [deep]
├── Task 19: Backward compatibility check with ml_initializer patterns [deep]
└── Task 20: Cleanup and final verification [quick]

Wave FINAL (After ALL tasks — independent review, 4 parallel):
├── Task F1: Dependency verification (ruff, build check) [quick]
├── Task F2: Code quality review (linter, type checking) [unspecified-high]
├── Task F3: Real manual QA (test all scenarios) [unspecified-high]
└── Task F4: Scope fidelity check (verify no creep) [deep]

Critical Path: Task 1 → Task 7 → Task 9 → Task 10 → Task 17 → F1-F4
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 8 (Waves 2 & 3)
```

### Dependency Matrix (abbreviated — show ALL tasks in your generated plan)

- **1-3**: — — 4-8, 1
- **4-8**: — — 7, 9-11, 2
- **7**: 4-6, 8 — 9, 3
- **9**: — — 10-11, 4
- **10**: 7, 9 — 11, 12-16, 4
- **11**: 9, 10 — 12-16, 17-19, 3
- **12-16**: — — 17-19, 4
- **17**: 9-11, 12-16 — 18-20, 3
- **18**: 17 — 19, 4
- **19**: 17 — 20, 4
- **20**: 17-19 — F1-F4, 4

> This is abbreviated for reference. YOUR generated plan must include the FULL matrix for ALL tasks.

### Agent Dispatch Summary

- **1**: **3** — T1 → `quick`, T2 → `quick`, T3 → `quick`
- **2**: **5** — T4 → `quick`, T5 → `quick`, T6 → `quick`, T7 → `quick`, T8 → `quick`
- **3**: **3** — T9 → `quick`, T10 → `deep`, T11 → `deep`
- **4**: **5** — T12 → `quick`, T13 → `quick`, T14 → `quick`, T15 → `quick`, T16 → `quick`
- **5**: **4** — T17 → `deep`, T18 → `deep`, T19 → `deep`, T20 → `quick`
- **FINAL**: **4** — F1 → `quick`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. Add Pydantic dependency to pyproject.toml

  **What to do**:
  - Add `pydantic>=2.0` to the `dependencies` section in `pyproject.toml`
  - Ensure it's in main dependencies, not optional dependencies
  - Add to both `cpu`, `gpu`, and `dev` optional-dependencies for consistency

  **Must NOT do**:
  - Do NOT add to `[project.optional-dependencies]` only (must be in main deps)
  - Do NOT pin to exact version (use `>=2.0` for flexibility)

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Simple dependency addition, file editing, minimal complexity
  - **Skills**: [] (no special skills needed)
    - Justification: Standard file editing and dependency management

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (blocks all other tasks)
  - **Blocks**: Tasks 2-8 (model definitions can't be verified until dep is installed)
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL - Be Exhaustive):

  > The executor has NO context from your interview. References are their ONLY guide.
  > Each reference must answer: "What should I look at and WHY?"

  **Pattern References** (existing code to follow):
  - `pyproject.toml:14-38` - Dependencies section structure (add new line to list)

  **API/Type References** (contracts to implement against):
  - Pydantic documentation: https://docs.pydantic.dev/latest/install/ - Add to dependencies

  **Test References** (testing patterns to follow):
  - No test files for dependency changes (verify installation)

  **External References** (libraries and frameworks):
  - Pydantic installation: `pip install pydantic` or `uv add pydantic`

  **WHY Each Reference Matters** (explain the relevance):
  - `pyproject.toml:14-38` - Show exact location and format for adding dependencies

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY** — No human action permitted.
  > Every criterion MUST be verifiable by running a command or using a tool.

  **If TDD (tests enabled):**
  - [ ] Dependency added to pyproject.toml: `pydantic>=2.0` in dependencies list
  - [ ] `uv sync` command succeeds without errors
  - [ ] `python -c "import pydantic; print(pydantic.__version__)"` succeeds

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify Pydantic installation
    Tool: Bash (python + uv)
    Preconditions: Fresh clone, no pydantic installed
    Steps:
      1. Run `uv add pydantic` or verify pyproject.toml has pydantic
      2. Run `uv sync` to install dependencies
      3. Run `python -c "import pydantic; print(pydantic.__version__)"`
    Expected Result: Command outputs Pydantic version (e.g., "2.10.0") without errors
    Failure Indicators: ImportError, uv sync errors, version parsing errors
    Evidence: .sisyphus/evidence/task-1-pydantic-install.txt

  Scenario: Verify dependency in pyproject.toml
    Tool: Bash (grep)
    Preconditions: pyproject.toml updated
    Steps:
      1. Run `grep "pydantic" pyproject.toml`
    Expected Result: Output shows `pydantic>=2.0` in dependencies
    Failure Indicators: No match, wrong version constraint, in wrong section
    Evidence: .sisyphus/evidence/task-1-dependency-check.txt
  ```

  **Evidence to Capture**:
  - [ ] `uv sync` output showing pydantic installation
  - [ ] Python import test output
  - [ ] grep output verifying dependency

  **Commit**: YES (groups with 1)
  - Message: `chore(deps): add pydantic>=2.0 to dependencies`
  - Files: `pyproject.toml`
  - Pre-commit: `uv sync && python -c "import pydantic"`

- [ ] 2. Create test file structure and pytest configuration

  **What to do**:
  - Create `tests/test_load_config.py` if it doesn't exist
  - Create `tests/fixtures/configs/` directory for test TOML files
  - Add test fixtures directory with these specific TOML files:
    * `valid.toml` - Complete valid config with all sections
    * `missing_target.toml` - Missing target section
    * `missing_cal_settings.toml` - Missing cal_settings
    * `invalid_type.toml` - Invalid type error (cutoff_value as string)
    * `optional_sections.toml` - With rnucleus, server_settings, model_params
    * `class_weight.toml` - For class_weight conversion test
    * `empty_spectral_term.toml` - Empty spectral_term list validation
    * `realistic.toml` - Integration test with realistic config
  - Create test fixtures using SimpleNamespace format (before refactoring) as baseline
  - Add pytest configuration if needed (conftest.py or pytest.ini)

  **Must NOT do**:
  - Do NOT modify existing test infrastructure

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: File creation, test scaffolding, fixture setup
  - **Skills**: []
    - Justification: Standard file operations and test setup

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4-20 (test file and fixtures needed before model definition and implementation)
  - **Blocked By**: Task 1 (pyproject.toml needs to be updated first)

  **References** (CRITICAL - Be Exhaustive):
  - `pyproject.toml:55-72` - Dev dependencies include pytest
  - `src/graspkit/data_IO/processing_data_loader.py:106-238` - Config structure for fixture creation
  - `tests/` directory - Look at existing test file patterns

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: `tests/test_load_config.py`
  - [ ] Fixtures directory created: `tests/fixtures/configs/`
  - [ ] All 8 test fixture TOML files created
  - [ ] `pytest tests/test_load_config.py --collect-only` shows collected tests

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify test file structure
    Tool: Bash (ls, test)
    Preconditions: File creation completed
    Steps:
      1. Run `ls -la tests/test_load_config.py`
      2. Run `ls -la tests/fixtures/configs/`
      3. Count fixture files: `ls tests/fixtures/configs/ | wc -l` (should be 8)
      4. Run `pytest tests/test_load_config.py --collect-only`
    Expected Result: Test file exists, fixtures directory exists, 8 fixture files created, pytest collects tests
    Failure Indicators: File not found, directory not created, wrong number of fixtures, pytest errors
    Evidence: .sisyphus/evidence/task-2-test-structure.txt
  ```

  **Evidence to Capture**:
  - [ ] Directory listing showing test file
  - [ ] Directory listing showing fixtures directory with 8 files
  - [ ] pytest collect output

  **Commit**: YES (groups with 1)
  - Message: `test(test): add test file structure and 8 fixture TOML files`
  - Files: `tests/test_load_config.py`, `tests/fixtures/configs/*.toml`
  - Pre-commit: none

- [ ] 3. Write failing tests for load_config behavior

  **What to do**:
  - Write RED tests for current load_config behavior (before refactoring)
  - Test cases: valid config, missing required fields, type errors, optional sections, class_weight conversion, cal_path dynamic attributes
  - Ensure tests FAIL with current implementation due to missing test cases (not bugs)
  - Use pytest fixtures for test TOML files

  **Test Cases to Cover**:
  1. Valid config with all sections
  2. Missing required section (target)
  3. Missing required field in cal_settings (cutoff_value)
  4. Invalid type (cutoff_value as string)
  5. Optional rnucleus section present and valid
  6. Optional sections absent
  7. class_weight with string keys converted to int
  8. spectral_term empty list (validation)
  9. cal_path dynamic attribute addition
  10. Path type conversion (root_path)

  **Test Fixtures to Create** (must be created before tests run):
  - `tests/fixtures/configs/valid.toml` - Complete valid config
  - `tests/fixtures/configs/missing_target.toml` - Missing target section
  - `tests/fixtures/configs/missing_cal_settings.toml` - Missing cal_settings
  - `tests/fixtures/configs/invalid_type.toml` - Invalid type error
  - `tests/fixtures/configs/optional_sections.toml` - With rnucleus, server_settings, model_params
  - `tests/fixtures/configs/class_weight.toml` - For class_weight conversion test
  - `tests/fixtures/configs/empty_spectral_term.toml` - Empty spectral_term list validation
  - `tests/fixtures/configs/realistic.toml` - Integration test with realistic config

  **Must NOT do**:
  - Do NOT write tests that rely on internal implementation details (SimpleNamespace vs Pydantic)
  - Do NOT test functionality not in current scope

  **Must NOT do**:
  - Do NOT write tests that rely on internal implementation details (SimpleNamespace vs Pydantic)
  - Do NOT test functionality not in current scope

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Test writing, understanding current behavior, pytest patterns
  - **Skills**: []
    - Justification: Standard pytest test writing

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: None (can start after Task 2)
  - **Blocked By**: Task 2 (test file structure must exist)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:106-238` - Current load_config implementation
  - `src/graspkit/ml_module/ml_initializer.py:31-123` - How config is used
  - `_validate_config_data` function - Validation logic to replicate in tests

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file written with all test cases
  - [ ] `pytest tests/test_load_config.py -v` runs and shows tests FAILING
  - [ ] Each test has clear assertion and expected behavior

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify failing tests (RED phase)
    Tool: Bash (pytest)
    Preconditions: Test file written, current implementation unchanged
    Steps:
      1. Run `pytest tests/test_load_config.py -v`
    Expected Result: Tests run and most FAIL (due to new test cases for edge cases not yet covered)
    Failure Indicators: Tests pass immediately (missing test cases), pytest crashes
    Evidence: .sisyphus/evidence/task-3-failing-tests.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing test failures (RED state)

  **Commit**: YES (groups with 1)
  - Message: `test(test): add failing tests for load_config behavior (RED)`
  - Files: `tests/test_load_config.py`, `tests/fixtures/configs/`
  - Pre-commit: `pytest tests/test_load_config.py`

- [ ] 4. Define Target, CalPath, MlConfig Pydantic models

  **What to do**:
  - Create `src/graspkit/data_IO/config_models.py` file
  - Define `Target` BaseModel with fields: `atom` (str), `conf` (str), `full_CSFs_set_file` (str)
  - Define `CalPath` BaseModel with `ConfigDict(extra='allow')` for dynamic attributes
  - Define `MlConfig` BaseModel with fields: `overfitting_threshold`, `underfitting_threshold`, `use_cpp_descriptor_generator`
  - Use Python 3.13 type hints: `str | int`, not `Optional[str]`

  **Must NOT do**:
  - Do NOT add fields not in original TOML structure
  - Do NOT use Pydantic V1 syntax

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: BaseModel definition, straightforward data modeling
  - **Skills**: []
    - Justification: Standard Pydantic model definition

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5-8)
  - **Blocks**: Task 7 (root Config model needs nested models)
  - **Blocked By**: Task 1 (Pydantic must be installed)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:188-198` - target section structure
  - `src/graspkit/ml_module/ml_initializer.py:70-76, 86-88` - cal_path usage patterns
  - Pydantic docs: https://docs.pydantic.dev/latest/concepts/models/

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] File created: `src/graspkit/data_IO/config_models.py`
  - [ ] `Target` model defined with required fields
  - [ ] `CalPath` model defined with `extra='allow'`
  - [ ] `MlConfig` model defined
  - [ ] `python -c "from graspkit.data_IO.config_models import Target, CalPath, MlConfig"` succeeds

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify model definitions import
    Tool: Bash (python)
    Preconditions: File created
    Steps:
      1. Run `python -c "from graspkit.data_IO.config_models import Target, CalPath, MlConfig; print('OK')"`
    Expected Result: Command outputs "OK" without errors
    Failure Indicators: ImportError, SyntaxError, validation errors on model definition
    Evidence: .sisyphus/evidence/task-4-models-import.txt

  Scenario: Verify CalPath allows extra fields
    Tool: Bash (python)
    Preconditions: CalPath model defined
    Steps:
      1. Run `python -c "from graspkit.data_IO.config_models import CalPath; cp = CalPath(cal_settings=None); cp.custom_field = 'test'; print(hasattr(cp, 'custom_field'))"`
    Expected Result: Command outputs "True" (extra field successfully added)
    Failure Indicators: ValidationError, False output
    Evidence: .sisyphus/evidence/task-4-calpath-extra.txt
  ```

  **Evidence to Capture**:
  - [ ] Python import test output
  - [ ] Extra field test output

  **Commit**: YES (groups with 5-8)
  - Message: `feat(data-io): add Target, CalPath, MlConfig Pydantic models`
  - Files: `src/graspkit/data_IO/config_models.py`
  - Pre-commit: `python -c "from graspkit.data_IO.config_models import ..."`

- [ ] 5. Define CalSettings, Rnucleus, ServerSettings models

  **What to do**:
  - Define `CalSettings` BaseModel with all required fields and optional fields
  - Fields: `root_path` (Path), `cal_loop_num` (int), `cutoff_value` (float), `sampling_ratio` (float), `expansion_ratio` (float), `difference` (int), `spectral_term` (list[str] | None)
  - Define `Rnucleus` BaseModel (optional): `atomic_number` (int), `mass_number` (int)
  - Define `ServerSettings` BaseModel (optional): `tasks_per_node` (int | None), `cpu_threads` (int | None)
  - Add field_validator for cutoff_value > 0
  - Add field_validator for sampling_ratio in (0, 1]
  - Use Python 3.13 type hints

  **Must NOT do**:
  - Do NOT add validation logic that differs from current behavior without approval

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: BaseModel definition with validators
  - **Skills**: []
    - Justification: Standard Pydantic field_validator usage

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6-8)
  - **Blocks**: Task 7 (root Config model needs these nested models)
  - **Blocked By**: Task 1 (Pydantic must be installed)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:134-159` - CalSettings fields and type conversions
  - `src/graspkit/data_IO/processing_data_loader.py:230-234` - Rnucleus validation
  - `src/graspkit/data_IO/processing_data_loader.py:224-228` - ServerSettings validation
  - Pydantic docs: https://docs.pydantic.dev/latest/concepts/validators/

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `CalSettings` model defined with all fields
  - [ ] `Rnucleus` model defined as optional
  - [ ] `ServerSettings` model defined as optional
  - [ ] Field validators added for numeric constraints
  - [ ] `python -c "from graspkit.data_IO.config_models import CalSettings, Rnucleus, ServerSettings"` succeeds

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify CalSettings validators
    Tool: Bash (python)
    Preconditions: CalSettings model defined
    Steps:
      1. Run `python -c "from graspkit.data_IO.config_models import CalSettings; from pydantic import ValidationError; try: CalSettings(cutoff_value=-1.0); except ValidationError as e: print('PASS')"`
    Expected Result: Command outputs "PASS" (validation error raised for cutoff_value <= 0)
    Failure Indicators: No ValidationError, validator not triggered
    Evidence: .sisyphus/evidence/task-5-calsettings-validator.txt

  Scenario: Verify optional models work
    Tool: Bash (python)
    Preconditions: Optional models defined
    Steps:
      1. Run `python -c "from graspkit.data_IO.config_models import Rnucleus; r = Rnucleus(); print('OK')"`
    Expected Result: Command outputs "OK" (empty Rnucleus accepted)
    Failure Indicators: ValidationError for empty model
    Evidence: .sisyphus/evidence/task-5-optional-models.txt
  ```

  **Evidence to Capture**:
  - [ ] Validator test output
  - [ ] Optional model test output

  **Commit**: YES (groups with 5-8)
  - Message: `feat(data-io): add CalSettings, Rnucleus, ServerSettings Pydantic models`
  - Files: `src/graspkit/data_IO/config_models.py`
  - Pre-commit: none

- [ ] 6. Define ModelParams Pydantic model

  **What to do**:
  - Define `ModelParams` BaseModel (optional)
  - Fields: `n_estimators` (int | None), `random_state` (int | None), `class_weight` (dict[int, float] | None)
  - Add field_validator for class_weight: convert string keys to int keys
  - Use `field_validator('class_weight', mode='before')` for key transformation
  - Use Python 3.13 type hints

  **Implementation Detail**:
  - class_weight validator should iterate dict items and create new dict with int(keys)
  - Example: `{"0": 1.0, "1": 2.0}` → `{0: 1.0, 1: 2.0}`

  **Must NOT do**:
  - Do NOT change the conversion logic behavior (str→int keys must be preserved)

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: BaseModel with before validator for dict key transformation
  - **Skills**: []
    - Justification: Standard Pydantic field_validator usage

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 7-8)
  - **Blocks**: Task 7 (root Config model needs this nested model)
  - **Blocked By**: Task 1 (Pydantic must be installed)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:160-178` - class_weight conversion logic

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `ModelParams` model defined
  - [ ] class_weight field validator converts string keys to int keys
  - [ ] Optional fields work correctly

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify class_weight key conversion
    Tool: Bash (python)
    Preconditions: ModelParams model defined
    Steps:
      1. Run `python -c "from graspkit.data_IO.config_models import ModelParams; mp = ModelParams(class_weight={'0': 1.0}); print(type(list(mp.class_weight.keys())[0]))"`
    Expected Result: Command outputs "<class 'int'>" (string key '0' converted to int 0)
    Failure Indicators: KeyError, TypeError, keys remain strings
    Evidence: .sisyphus/evidence/task-6-classweight-keys.txt
  ```

  **Evidence to Capture**:
  - [ ] Python test output showing int keys

  **Commit**: YES (groups with 5-8)
  - Message: `feat(data-io): add ModelParams Pydantic model`
  - Files: `src/graspkit/data_IO/config_models.py`
  - Pre-commit: none

- [ ] 7. Define root Config model with all sections

  **What to do**:
  - Define `Config` root BaseModel
  - Fields: `target` (Target), `cal_settings` (CalSettings), `rnucleus` (Rnucleus | None), `server_settings` (ServerSettings | None), `model_params` (ModelParams | None), `ml_config` (MlConfig), `cal_path` (CalPath | None)
  - All sections except `cal_path` are required (no default)
  - `cal_path` is optional (defaults to None or CalPath())

  **Must NOT do**:
  - Do NOT add new sections
  - Do NOT change required/optional status without approval

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Root model combining all nested models
  - **Skills**: []
    - Justification: Standard Pydantic model composition

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4-6, 8)
  - **Blocks**: Tasks 9-11 (load_config refactor needs this root model)
  - **Blocked By**: Tasks 4-6 (all nested models must be defined first)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:106-123` - load_config function return structure
  - All model definitions from Tasks 4-6

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `Config` model defined with all sections
  - [ ] Nested models properly composed
  - [ ] Optional sections work
  - [ ] `python -c "from graspkit.data_IO.config_models import Config; c = Config(...)"` succeeds

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify root Config model composition
    Tool: Bash (python)
    Preconditions: All nested models defined
    Steps:
      1. Run `python -c "from graspkit.data_IO.config_models import Config; print(Config.model_fields.keys())"`
    Expected Result: Command shows all 7 sections: target, cal_settings, rnucleus, server_settings, model_params, ml_config, cal_path
    Failure Indicators: Missing fields, wrong types, composition errors
    Evidence: .sisyphus/evidence/task-7-root-model.txt
  ```

  **Evidence to Capture**:
  - [ ] Python test output showing all model fields

  **Commit**: YES (groups with 5-8)
  - Message: `feat(data-io): add root Config Pydantic model`
  - Files: `src/graspkit/data_IO/config_models.py`
  - Pre-commit: none

- [ ] 8. Create config_models.py module for all models

  **What to do**:
  - Ensure `src/graspkit/data_IO/config_models.py` has proper module header
  - Add docstring with @Id, @date, @author fields
  - Export all models: `Config`, `Target`, `CalSettings`, `Rnucleus`, `ServerSettings`, `ModelParams`, `MlConfig`, `CalPath`
  - Add `__all__` list for explicit exports

  **Must NOT do**:
  - Do NOT change model definitions from Tasks 4-7

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Module setup, exports, docstring
  - **Skills**: []
    - Justification: Standard Python module conventions

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4-7)
  - **Blocks**: Tasks 9-11 (load_config needs to import these models)
  - **Blocked By**: Tasks 4-7 (all models must be defined)

  **References** (CRITICAL - Be Exhaustive):
  - `AGENTS.md:1-40` - Module header format (@Id, @date, @author)
  - `src/graspkit/data_IO/__init__.py` - Module export patterns

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Module has proper header with @Id, @date, @author
  - [ ] All models exported in `__all__`
  - [ ] `from graspkit.data_IO.config_models import Config, Target, ...` succeeds

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify module exports
    Tool: Bash (python)
    Preconditions: Module file created
    Steps:
      1. Run `python -c "from graspkit.data_IO.config_models import Config, Target, CalSettings, Rnucleus, ServerSettings, ModelParams, MlConfig, CalPath; print('All exports OK')"`
    Expected Result: Command outputs "All exports OK" without ImportError
    Failure Indicators: ImportError for any model
    Evidence: .sisyphus/evidence/task-8-module-exports.txt
  ```

  **Evidence to Capture**:
  - [ ] Python import test output

  **Commit**: YES (groups with 5-8)
  - Message: `feat(data-io): add config_models.py module for all models`
  - Files: `src/graspkit/data_IO/config_models.py`
  - Pre-commit: none

- [ ] 9. Update load_config imports and add Pydantic models

  **What to do**:
  - Update `load_config` function imports in `src/graspkit/data_IO/processing_data_loader.py`
  - Import `Config` and all nested models from `config_models`
  - Remove `SimpleNamespace` import (marked for deletion in Task 13)
  - Keep existing imports: `Path`, `Any`, `rtoml`, `CSFs`

  **Must NOT do**:
  - Do NOT remove `CSFs` import (used elsewhere in file)
  - Do NOT change function signature parameters

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Import updates, minimal changes
  - **Skills**: []
    - Justification: Standard import management

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 10-11)
  - **Blocks**: Task 10 (load_config refactor needs imports)
  - **Blocked By**: Task 8 (config_models.py must exist)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:1-22` - Current imports section
  - `src/graspkit/data_IO/config_models.py` - New models to import

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `from graspkit.data_IO.config_models import Config` added
  - [ ] `SimpleNamespace` import still present (will be removed in Task 13)
  - [ ] Other imports unchanged
  - [ ] `python -c "from graspkit.data_IO.processing_data_loader import load_config; print('Import OK')"` succeeds

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify imports are correct
    Tool: Bash (python)
    Preconditions: Import statements added
    Steps:
      1. Run `python -c "from graspkit.data_IO.processing_data_loader import load_config; import sys; sys.path.insert(0, 'src/graspkit/data_IO'); from config_models import Config; print('OK')"`
    Expected Result: Command outputs "OK" without ImportError
    Failure Indicators: ImportError for Config or load_config
    Evidence: .sisyphus/evidence/task-9-imports.txt
  ```

  **Evidence to Capture**:
  - [ ] Python import test output

  **Commit**: YES (groups with 10-11)
  - Message: `refactor(data-io): update load_config imports for Pydantic models`
  - Files: `src/graspkit/data_IO/processing_data_loader.py`
  - Pre-commit: `python -c "from graspkit.data_IO.config_models import Config"`

- [ ] 10. Implement load_config with model_validate (keep old code)

  **What to do**:
  - Update `load_config` function signature return type from `SimpleNamespace` to `Config`
  - Replace `_dict_to_namespace(processed)` call with `Config.model_validate(processed)`
  - Keep `_process_config_data` and `_validate_config_data` calls temporarily (will be integrated in Task 11)
  - Keep all error handling and logging logic
  - Ensure `load_config` still returns a Pydantic Config object

  **Must NOT do**:
  - Do NOT delete helper functions yet (will be removed in Tasks 12-15)
  - Do NOT change error messages
  - Do NOT change function name or parameter names

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `deep`
    - Reason: Core refactoring, understanding existing data flow, maintaining backward compatibility
  - **Skills**: []
    - Justification: Requires careful analysis of existing code patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 9, 11)
  - **Blocks**: Task 11 (needs this refactor complete before validation integration)
  - **Blocked By**: Task 9 (imports must be updated)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:106-122` - Current load_config implementation
  - Pydantic docs: https://docs.pydantic.dev/latest/concepts/models/#helper-functions - model_validate usage

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `load_config` return type updated to `Config`
  - [ ] `_dict_to_namespace` call replaced with `Config.model_validate`
  - [ ] Old helper functions still present (temporary)
  - [ ] Function compiles without syntax errors
  - [ ] Existing tests still pass (no behavior change yet)

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify Pydantic model can load valid TOML
    Tool: Bash (python)
    Preconditions: Valid fixture file created in Task 2
    Steps:
      1. Run `python -c "from graspkit.data_IO.processing_data_loader import load_config; cfg = load_config('tests/fixtures/configs/valid.toml'); print(type(cfg)); print(cfg.target.atom); print(cfg.cal_settings.cutoff_value)"`
    Expected Result: Command outputs Pydantic model type, atom value, cutoff_value value
    Failure Indicators: ImportError, ValidationError, AttributeError
    Evidence: .sisyphus/evidence/task-10-load-valid-toml.txt
  ```

  **Evidence to Capture**:
  - [ ] Python test output showing successful load with Pydantic model

  **Commit**: YES (groups with 10-11)
  - Message: `refactor(data-io): refactor load_config to use Pydantic model_validate`
  - Files: `src/graspkit/data_IO/processing_data_loader.py`
  - Pre-commit: `pytest tests/test_load_config.py`

- [ ] 11. Integrate validation logic via field_validator

  **What to do**:
  - Move validation logic from `_validate_config_data` to Pydantic `field_validator` decorators in `config_models.py`
  - Implement validators for:
    - Required sections: target, cal_settings, ml_config
    - Required fields in target: atom, conf
    - Required fields in cal_settings: root_path, cal_loop_num, cutoff_value, sampling_ratio
    - Numeric constraints: cutoff_value > 0, sampling_ratio in (0, 1], atomic_number > 0
    - spectral_term: non-empty list if present
  - Use `@model_validator` for top-level section validation
  - Keep error messages in Chinese to match current behavior

  **Must NOT do**:
  - Do NOT change validation logic without approval
  - Do NOT modernize error messages (keep Chinese, same format)

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `deep`
    - Reason: Validation logic migration, understanding existing validation rules, Pydantic validators
  - **Skills**: []
    - Justification: Requires understanding of both existing validation and Pydantic validator API

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10)
  - **Blocks**: Tasks 12-16 (cleanup needs validation to be integrated)
  - **Blocked By**: Task 10 (load_config refactor must be complete)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:185-239` - _validate_config_data function
  - Pydantic docs: https://docs.pydantic.dev/latest/concepts/validators/

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] All validation logic moved to field_validator decorators
  - [ ] _validate_config_data function still present (will be removed in Task 15)
  - [ ] Validation errors produce same messages in Chinese
  - [ ] Tests pass for all validation scenarios

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify required section validation
    Tool: Bash (python)
    Preconditions: Validators integrated
    Steps:
      1. Run `python -c "from pydantic import ValidationError; from graspkit.data_IO.config_models import Config; try: Config(cal_settings={'root_path': '/tmp', 'cal_loop_num': 1}); except ValidationError as e: print(e.errors()[0]['msg'])"`
    Expected Result: Command outputs Chinese error message about missing required section (target)
    Failure Indicators: ValidationError with wrong message, no error raised
    Evidence: .sisyphus/evidence/task-11-validation-errors.txt
  ```

  **Evidence to Capture**:
  - [ ] Validation error test output in Chinese

  **Commit**: YES (groups with 10-11)
  - Message: `refactor(data-io): integrate validation logic via Pydantic field_validator`
  - Files: `src/graspkit/data_IO/config_models.py`
  - Pre-commit: `pytest tests/test_load_config.py`

- [ ] 12. Remove _dict_to_namespace function

  **What to do**:
  - Delete `_dict_to_namespace` function from `src/graspkit/data_IO/processing_data_loader.py` (lines 124-131)
  - This function is no longer needed since Pydantic handles nested models automatically

  **Must NOT do**:
  - Do NOT delete any other functions (unless they're also obsolete)

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Function deletion, straightforward removal
  - **Skills**: []
    - Justification: Simple code removal

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 13-16)
  - **Blocks**: Task 17 (tests need all cleanup done)
  - **Blocked By**: Task 11 (validation must be integrated first)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:124-131` - _dict_to_namespace function location

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `_dict_to_namespace` function deleted
  - [ ] `grep -n "_dict_to_namespace" src/graspkit/data_IO/processing_data_loader.py` returns no results
  - [ ] Tests still pass (function was only used by load_config)

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify _dict_to_namespace removed
    Tool: Bash (grep)
    Preconditions: Function deleted
    Steps:
      1. Run `grep -n "_dict_to_namespace" src/graspkit/data_IO/processing_data_loader.py`
    Expected Result: Command returns nothing (no matches found)
    Failure Indicators: Function still exists, grep returns line number
    Evidence: .sisyphus/evidence/task-12-removed-function.txt
  ```

  **Evidence to Capture**:
  - [ ] grep output showing no matches

  **Commit**: YES (groups with 12-16)
  - Message: `refactor(data-io): remove _dict_to_namespace function`
  - Files: `src/graspkit/data_IO/processing_data_loader.py`
  - Pre-commit: none

- [ ] 13. Remove SimpleNamespace import

  **What to do**:
  - Remove `from types import SimpleNamespace` from line 10
  - This import is no longer needed since we use Pydantic models

  **Must NOT do**:
  - Do NOT remove other imports

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Import removal, straightforward
  - **Skills**: []
    - Justification: Simple import removal

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 12, 14-16)
  - **Blocks**: Task 17 (tests need all cleanup done)
  - **Blocked By**: Task 12 (ensure _dict_to_namespace removed first)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:10` - SimpleNamespace import location

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `from types import SimpleNamespace` removed from line 10
  - [ ] `grep "SimpleNamespace" src/graspkit/data_IO/processing_data_loader.py` returns no results
  - [ ] `python -c "from graspkit.data_IO.processing_data_loader import load_config"` succeeds

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify SimpleNamespace import removed
    Tool: Bash (grep)
    Preconditions: Import removed
    Steps:
      1. Run `grep "SimpleNamespace" src/graspkit/data_IO/processing_data_loader.py`
    Expected Result: Command returns nothing (no matches found)
    Failure Indicators: Import still exists, grep returns line number
    Evidence: .sisyphus/evidence/task-13-removed-import.txt
  ```

  **Evidence to Capture**:
  - [ ] grep output showing no matches

  **Commit**: YES (groups with 12-16)
  - Message: `refactor(data-io): remove SimpleNamespace import`
  - Files: `src/graspkit/data_IO/processing_data_loader.py`
  - Pre-commit: none

- [ ] 14. Remove _process_config_data function

  **What to do**:
  - Delete `_process_config_data` function from `src/graspkit/data_IO/processing_data_loader.py` (lines 134-182)
  - Type conversion logic is now handled by Pydantic field validators in config_models.py

  **Must NOT do**:
  - Do NOT delete validation logic (moved to field_validator)
  - Do NOT delete other functions

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Function deletion, type conversion moved to Pydantic
  - **Skills**: []
    - Justification: Pydantic handles type conversion automatically

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 12-13, 15-16)
  - **Blocks**: Task 17 (tests need all cleanup done)
  - **Blocked By**: Task 11 (validation must be integrated first)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:134-182` - _process_config_data function location

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `_process_config_data` function deleted
  - [ ] `grep -n "_process_config_data" src/graspkit/data_IO/processing_data_loader.py` returns no results
  - [ ] Tests still pass (type conversion now via Pydantic)

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify _process_config_data removed
    Tool: Bash (grep)
    Preconditions: Function deleted
    Steps:
      1. Run `grep -n "_process_config_data" src/graspkit/data_IO/processing_data_loader.py`
    Expected Result: Command returns nothing (no matches found)
    Failure Indicators: Function still exists, grep returns line number
    Evidence: .sisyphus/evidence/task-14-removed-function.txt
  ```

  **Evidence to Capture**:
  - [ ] grep output showing no matches

  **Commit**: YES (groups with 12-16)
  - Message: `refactor(data-io): remove _process_config_data function`
  - Files: `src/graspkit/data_IO/processing_data_loader.py`
  - Pre-commit: none

- [ ] 15. Remove _validate_config_data function

  **What to do**:
  - Delete `_validate_config_data` function from `src/graspkit/data_IO/processing_data_loader.py` (lines 185-239)
  - Validation logic is now in Pydantic field decorators in config_models.py

  **Must NOT do**:
  - Do NOT remove validation logic (moved to field_validator)

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Function deletion, validation moved to Pydantic
  - **Skills**: []
    - Justification: Pydantic field validators replace custom validation function

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 12-14, 16)
  - **Blocks**: Task 17 (tests need all cleanup done)
  - **Blocked By**: Task 11 (validation must be integrated first)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/data_IO/processing_data_loader.py:185-239` - _validate_config_data function location

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `_validate_config_data` function deleted
  - [ ] `grep -n "_validate_config_data" src/graspkit/data_IO/processing_data_loader.py` returns no results
  - [ ] Tests still pass (validation now via Pydantic)

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify _validate_config_data removed
    Tool: Bash (grep)
    Preconditions: Function deleted
    Steps:
      1. Run `grep -n "_validate_config_data" src/graspkit/data_IO/processing_data_loader.py`
    Expected Result: Command returns nothing (no matches found)
    Failure Indicators: Function still exists, grep returns line number
    Evidence: .sisyphus/evidence/task-15-removed-function.txt
  ```

  **Evidence to Capture**:
  - [ ] grep output showing no matches

  **Commit**: YES (groups with 12-16)
  - Message: `refactor(data-io): remove _validate_config_data function`
  - Files: `src/graspkit/data_IO/processing_data_loader.py`
  - Pre-commit: none

- [ ] 16. Update type hints to Python 3.13 style

  **What to do**:
  - Review all type hints in `src/graspkit/data_IO/processing_data_loader.py`
  - Replace `Optional[str]` with `str | None`
  - Replace `Union[str, Path]` with `str | Path`
  - Replace `List[str]` with `list[str]`
  - Replace `Dict[str, Any]` with `dict[str, Any]`
  - Ensure all function return types use `|` operator for unions

  **Files to update**:
  - `load_config` function signature and any remaining helper functions
  - Other functions in the file should also be checked for type hints

  **Must NOT do**:
  - Do NOT use `from typing import Optional, Union, List, Dict` where not needed
  - Do NOT change behavior, only type hint syntax

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Type hint modernization to Python 3.13 style
  - **Skills**: []
    - Justification: Follow AGENTS.md guidelines for Python 3.13 type hints

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 12-15)
  - **Blocks**: Task 17 (tests need all updates done)
  - **Blocked By**: Tasks 12-15 (cleanup should happen first)

  **References** (CRITICAL - Be Exhaustive):
  - `AGENTS.md:1-40` - Python 3.13 type hint guidelines
  - `src/graspkit/data_IO/processing_data_loader.py:1-238` - All type hints in file

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] All Optional[T] replaced with T | None
  - [ ] All Union[T1, T2] replaced with T1 | T2
  - [ ] `from typing` imports updated (remove Optional, Union, List, Dict if not used elsewhere)
  - [ ] No `ruff check .` errors for type hints
  - [ ] `mypy` passes

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify type hints updated to Python 3.13 style
    Tool: Bash (ruff, mypy)
    Preconditions: Type hints updated
    Steps:
      1. Run `ruff check src/graspkit/data_IO/processing_data_loader.py`
      2. Run `mypy src/graspkit/data_IO/processing_data_loader.py`
    Expected Result: Both commands complete without errors
    Failure Indicators: Ruff errors about type hints, mypy errors
    Evidence: .sisyphus/evidence/task-16-typehints.txt
  ```

  **Evidence to Capture**:
  - [ ] ruff check output
  - [ ] mypy check output

  **Commit**: YES (groups with 12-16)
  - Message: `refactor(data-io): update type hints to Python 3.13 style`
  - Files: `src/graspkit/data_IO/processing_data_loader.py`
  - Pre-commit: `ruff check src/graspkit/data_IO/processing_data_loader.py && mypy src/graspkit/data_IO/processing_data_loader.py`

- [ ] 17. Run all tests and verify behavior parity

  **What to do**:
  - Run `pytest tests/test_load_config.py -v` to ensure all tests pass
  - Verify that Pydantic implementation matches SimpleNamespace behavior
  - Check that all attribute access patterns work
  - Verify dynamic cal_path attributes can be added
  - Compare behavior before/after for edge cases

  **Test Scenarios to Verify**:
  1. Valid config with all sections
  2. Missing required section (target)
  3. Missing required field in cal_settings (cutoff_value)
  4. Invalid type (cutoff_value as string)
  5. Optional rnucleus section present and valid
  6. Optional sections absent
  7. class_weight with string keys converted to int
  8. spectral_term empty list (validation)
  9. cal_path dynamic attribute addition
  10. Path type conversion (root_path)

  **Must NOT do**:
  - Do NOT modify tests without approval
  - Do NOT skip failing tests

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `deep`
    - Reason: Comprehensive test verification, behavior parity check
  - **Skills**: []
    - Justification: Standard pytest execution and verification

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (blocks Tasks 18-20)
  - **Blocks**: Tasks 18-20 (verification needs complete implementation)
  - **Blocked By**: Tasks 12-16 (all cleanup must be complete)

  **References** (CRITICAL - Be Exhaustive):
  - `tests/test_load_config.py` - All test cases written in Task 3

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `pytest tests/test_load_config.py -v` shows all tests PASS
  - [ ] All 10 test scenarios pass
  - [ ] No regressions from SimpleNamespace behavior
  - [ ] cal_path dynamic attributes work

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify all tests pass
    Tool: Bash (pytest)
    Preconditions: Implementation complete
    Steps:
      1. Run `pytest tests/test_load_config.py -v`
    Expected Result: All tests pass (PASSED), no failures or errors
    Failure Indicators: Any test FAIL, pytest errors
    Evidence: .sisyphus/evidence/task-17-all-tests-pass.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing all tests pass

  **Commit**: YES (groups with 17-20)
  - Message: `test(test): verify all tests pass and behavior parity`
  - Files: `tests/test_load_config.py`
  - Pre-commit: `pytest tests/test_load_config.py -v`

- [ ] 18. Integration test with actual TOML config

  **What to do**:
  - Create or use existing TOML config file for integration test
  - Test load_config with realistic config (similar to actual usage)
  - Verify all attribute access patterns from ml_initializer.py work
  - Test cal_path dynamic attribute addition scenario
  - Verify error messages match expectations

  **Must NOT do**:
  - Do NOT use mock config files only (test with realistic data)

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `deep`
    - Reason: Integration testing with actual usage patterns
  - **Skills**: []
    - Justification: Real-world scenario verification

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Task 17, 19-20)
  - **Blocks**: Task 19 (backward compat check)
  - **Blocked By**: Task 17 (unit tests must pass first)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/ml_module/ml_initializer.py:53-123` - Actual config usage patterns
  - Existing config files in project (if any) for realistic test data

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Integration test created or uses realistic config
  - [ ] All attribute access patterns from ml_initializer work
  - [ ] cal_path dynamic attributes can be added
  - [ ] Error messages as expected

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Integration test with realistic config
    Tool: Bash (python)
    Preconditions: Realistic TOML config created in Task 2
    Steps:
      1. Run `python -c "from graspkit.data_IO.processing_data_loader import load_config; cfg = load_config('tests/fixtures/configs/realistic.toml'); print(type(cfg)); print(hasattr(cfg, 'cal_settings')); print(hasattr(cfg.cal_settings, 'root_path'))"`
    Expected Result: Outputs Pydantic model type, True for hasattr, successful attribute access
    Failure Indicators: AttributeError, TypeError
    Evidence: .sisyphus/evidence/task-18-integration-test.txt
  ```

  **Evidence to Capture**:
  - [ ] Integration test output

  **Commit**: YES (groups with 17-20)
  - Message: `test(integration): integration test with realistic TOML config`
  - Files: `tests/test_load_config.py`
  - Pre-commit: `pytest tests/test_load_config.py -k integration`

- [ ] 19. Backward compatibility check with ml_initializer patterns

  **What to do**:
  - Verify that patterns in ml_initializer.py still work with Pydantic config
  - Test attribute access: `config.cal_settings.root_path`, `config.target.conf`, `config.cal_path.custom_field`
  - Test getattr with defaults: `getattr(config.server_settings, "cpu_threads", 16)`
  - Test hasattr checks: `hasattr(config, "rnucleus")`
  - Ensure no regressions in config usage

  **Must NOT do**:
  - Do NOT modify ml_initializer.py
  - Do NOT change config usage patterns

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `deep`
    - Reason: Backward compatibility verification, ensuring no regressions
  - **Skills**: []
    - Justification: Understanding existing usage patterns and verifying compatibility

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 17-18, 20)
  - **Blocks**: Task 20 (cleanup and final verification)
  - **Blocked By**: Task 18 (integration test needs to pass)

  **References** (CRITICAL - Be Exhaustive):
  - `src/graspkit/ml_module/ml_initializer.py:53-123` - All config usage patterns
  - `grep` results showing 45 usages of config across 6 files

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] All ml_initializer.py patterns still work
  - [ ] Attribute access works: `config.cal_settings.root_path`
  - [ ] Dynamic cal_path attributes work
  - [ ] getattr with defaults works
  - [ ] hasattr checks work
  - [ ] No AttributeError or TypeError

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Verify ml_initializer.py patterns work
    Tool: Bash (python)
    Preconditions: Pydantic config available, realistic.toml fixture exists from Task 2
    Steps:
      1. Run `python -c "from graspkit.data_IO.processing_data_loader import load_config; cfg = load_config('tests/fixtures/configs/realistic.toml'); root_path = cfg.cal_settings.root_path; print(type(root_path)); cal_path = type('CalPath')(); cfg.cal_path = cal_path; print(hasattr(cfg, 'cal_path'))"`
    Expected Result: Path type returned, hasattr returns True, no errors
    Failure Indicators: AttributeError, TypeError, False returned
    Evidence: .sisyphus/evidence/task-19-backward-compat.txt
  ```

  **Evidence to Capture**:
  - [ ] Python test output showing all patterns work

  **Commit**: YES (groups with 17-20)
  - Message: `test(compat): verify backward compatibility with ml_initializer patterns`
  - Files: `tests/test_load_config.py`
  - Pre-commit: none

- [ ] 20. Cleanup and final verification

  **What to do**:
  - Run `ruff check .` to ensure no linting issues
  - Run `mypy src/graspkit/data_IO/` to ensure type checking passes
  - Verify all old functions and imports are removed
  - Verify all new code follows AGENTS.md guidelines
  - Check for any remaining SimpleNamespace references
  - Final test run: `pytest tests/test_load_config.py -v`

  **Must NOT do**:
  - Do NOT make any behavioral changes
  - Do NOT add new features

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Final cleanup and verification
  - **Skills**: []
    - Justification: Standard code quality checks

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final task)
  - **Blocks**: F1-F4 (final verification wave)
  - **Blocked By**: Tasks 17-19 (all verification must complete)

  **References** (CRITICAL - Be Exhaustive):
  - `AGENTS.md` - All code style guidelines
  - Project root directory for running commands

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] `ruff check .` passes with 0 issues
  - [ ] `mypy src/graspkit/data_IO/` passes with 0 errors
  - [ ] `pytest tests/test_load_config.py -v` passes all tests
  - [ ] No SimpleNamespace references in codebase
  - [ ] All old functions removed

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: Final cleanup verification
    Tool: Bash (ruff, mypy, pytest)
    Preconditions: All implementation complete
    Steps:
      1. Run `ruff check .`
      2. Run `mypy src/graspkit/data_IO/`
      3. Run `pytest tests/test_load_config.py -v`
    Expected Result: All commands pass, 0 errors
    Failure Indicators: Any linting errors, type errors, test failures
    Evidence: .sisyphus/evidence/task-20-final-cleanup.txt
  ```

  **Evidence to Capture**:
  - [ ] ruff check output
  - [ ] mypy check output
  - [ ] pytest output

  **Commit**: YES (groups with 17-20)
  - Message: `refactor(data-io): cleanup and final verification`
  - Files: `src/graspkit/data_IO/processing_data_loader.py`, `tests/test_load_config.py`
  - Pre-commit: `ruff check . && mypy src/graspkit/data_IO/ && pytest tests/test_load_config.py -v`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [ ] F1. **Dependency Verification** — `quick`
  Run `uv sync` to verify Pydantic dependency installs correctly. Run `ruff check .` to ensure no linting issues. Run `python -m build` to verify package builds successfully.
  Output: `Pydantic [PASS/FAIL] | Ruff [PASS/FAIL] | Build [PASS/FAIL] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `ruff check . --fix` + `mypy src/graspkit/data_IO/`. Check for type hints, unused imports, code style. Verify no `as any` or `@ts-expect-error`.
  Output: `Ruff [PASS/FAIL] | Mypy [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `pytest` skill)
  Create test TOML files covering all scenarios: valid config, missing fields, type errors, optional sections, class_weight conversion, cal_path dynamic attributes. Run ALL test suites. Test integration with actual ml_initializer.py patterns.
  Output: `Tests [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1**: `chore(deps): add pydantic>=2.0 to dependencies` — pyproject.toml, `uv sync`
- **8**: `feat(data-io): add config_models.py with Pydantic models` — src/graspkit/data_IO/config_models.py
- **11**: `refactor(data-io): integrate validation in Pydantic models` — src/graspkit/data_IO/config_models.py
- **10**: `refactor(data-io): refactor load_config to use Pydantic` — src/graspkit/data_IO/processing_data_loader.py
- **16**: `refactor(data-io): remove legacy config helpers` — src/graspkit/data_IO/processing_data_loader.py

---

## Success Criteria

### Verification Commands
```bash
# Install dependencies
uv sync

# Run tests
pytest tests/test_load_config.py -v

# Type checking
mypy src/graspkit/data_IO/

# Linting
ruff check . --fix

# Build package
python -m build
```

### Final Checklist
- [ ] Pydantic V2 dependency added to pyproject.toml
- [ ] All Pydantic models defined (Target, CalSettings, Rnucleus, ServerSettings, ModelParams, MlConfig, CalPath, Config)
- [ ] load_config returns Pydantic Config model
- [ ] All validation logic integrated via field_validator
- [ ] _dict_to_namespace function deleted
- [ ] SimpleNamespace import removed
- [ ] All existing attribute access patterns work
- [ ] Dynamic cal_path attributes can be added
- [ ] All tests pass (TDD workflow completed)
- [ ] ruff check passes
- [ ] mypy passes (no type errors)
- [ ] Package builds successfully
- [ ] No regressions in ml_initializer usage
