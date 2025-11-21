@dependencies [TASK0, TASK1, TASK2, TASK3, TASK4, TASK5, TASK6, TASK7, TASK8, TASK9]
# Task: Automated Unit and Integration Testing

## Summary
Write and execute automated unit tests for all modified/new code. This complements manual testing (TASK11) and ensures code correctness at the unit level.

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **Test files:** All `.test.py` or `.test.js` files created in previous tasks
- **Scope:** Temperature fix, unit conversion, diagnostics, web JS
- **Framework:** Jest (JavaScript), pytest or unittest (Python)

## Complexity
Medium

## Dependencies
Depends on: [TASK0, TASK1, TASK2, TASK3, TASK4, TASK5, TASK6, TASK7, TASK8, TASK9]
Blocks: [TASKW]
Parallel with: [TASK11]

## Test Coverage Requirements
**Python (Desktop):**
- [ ] **A.1** Unit tests for temperature initialization (`test_particles.py`)
- [ ] **A.2** Unit tests for metric conversion functions (`test_unit_conversion.py`)
- [ ] **A.3** Integration test: load config, start sim, verify temperature in first progress (`test_simulation.py`)
- [ ] Tests for diagnostics widget updates (mock simulation data)
- [ ] Tests for preferences dialog save/load

**JavaScript (Web):**
- [ ] Unit tests for demo data generation
- [ ] Unit tests for color mapping functions
- [ ] Integration test: load demo → render → verify particles displayed
- [ ] (Optional) Selenium/Playwright for button interactions

## Acceptance Criteria
- [ ] All unit tests pass
- [ ] Coverage >80% for modified code
- [ ] Integration tests verify end-to-end flows
- [ ] No test failures or errors
- [ ] Test execution documented

## Code Review Checklist
- [ ] Tests are self-contained (mocks for external dependencies)
- [ ] Tests follow existing patterns (co-located with source)
- [ ] Clear test names and assertions
- [ ] No flaky tests (deterministic, no timing dependencies)
