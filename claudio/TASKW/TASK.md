@dependencies [TASK0, TASK1, TASK2, TASK3, TASK4, TASK5, TASK6, TASK7, TASK8, TASK9, TASK10, TASK11, TASK12]
# Task: Final Ω - System-Level Validation and Cohesion Verification

## Summary
**MANDATORY FINAL VALIDATION STEP**

Verify that the entire TDE-SPH GUI system is complete, coherent, and production-ready. This task ensures no requirements were forgotten, all components interact correctly, and the system meets all acceptance criteria from AI_PROMPT.md.

This is the **cohesion validation** layer that confirms the system works as a whole, not just as individual parts.

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **Validation scope:** ALL acceptance criteria from AI_PROMPT.md
- **Verification method:** Cross-check every criterion against implementation
- **Focus:** System integration, requirement traceability, production readiness

## Complexity
Medium (verification and documentation)

## Dependencies
Depends on: [TASK0, TASK1, TASK2, TASK3, TASK4, TASK5, TASK6, TASK7, TASK8, TASK9, TASK10, TASK11, TASK12]
Blocks: []
Parallel with: []

## Detailed Validation Steps

### 1. Requirement Traceability
For EVERY acceptance criterion in AI_PROMPT.md sections 1-7, verify:
- [ ] Criterion has corresponding implementation (file:line reference)
- [ ] Implementation tested (unit test or manual test result)
- [ ] No criterion was skipped or merged without justification

**Methodology:**
- Open AI_PROMPT.md section "Acceptance Criteria"
- For each criterion (C.1, C.2, ... A.1, ... T.1, ... etc.):
  - Find implementation in codebase
  - Verify test coverage exists
  - Document verification in checklist

### 2. System Integration Validation
Test complete workflows end-to-end:
- [ ] **Desktop workflow:** Config load → Simulation start → Real-time diagnostics → Web viewer → Export
- [ ] **Web workflow:** Launch → Load demo → Animate → Change visualization → Export screenshot
- [ ] **Bug fix workflow:** Verify temperature data flows correctly (ParticleSystem → Simulation → GUI)
- [ ] **Unit conversion workflow:** Toggle units → Verify all displays update → Toggle back
- [ ] **Preferences workflow:** Change settings → Restart app → Verify persistence

### 3. Data Flow Verification
Trace critical data flows through system:
- [ ] Temperature: EOS computes → Simulation stores → SimulationThread emits → GUI displays
- [ ] Particle stats: Simulation computes → Thread emits → Diagnostics updates
- [ ] Unit conversion: Config provides factors → Conversion applies → Display shows physical units
- [ ] Web data: File/demo loads → Parser extracts → Visualizer renders → User sees particles

### 4. Error Handling Verification
For each error scenario (E.1-E.8), verify:
- [ ] Error is caught (try/except or validation)
- [ ] User sees clear error message (not just console log)
- [ ] System recovers gracefully (no crash, UI state reset)
- [ ] Documented in code or user manual

### 5. Performance Validation
Measure actual performance against targets:
- [ ] Desktop GUI: Input lag <100ms (test button clicks, slider drags)
- [ ] Web rendering: 30+ fps for <100k particles (measure with FPS counter)
- [ ] Diagnostics update: <50ms to refresh (measure signal → display time)
- [ ] Memory usage: Reasonable for dataset size (monitor during simulation)

**Document measurements:**
- Actual fps: ___
- Input lag: ___
- Memory usage: ___

### 6. Code Quality Review
Global code quality checks:
- [ ] No placeholder code remains (search for "TODO", "FIXME", "NotImplemented")
- [ ] All hasattr/getattr defensive patterns removed (for temperature and similar bugs)
- [ ] No debug print statements in production code
- [ ] All imports used (no dead imports)
- [ ] Type hints present (Python 3.9+ style)
- [ ] Docstrings complete (NumPy style)

### 7. Documentation Completeness
- [ ] All new functions have docstrings
- [ ] Complex algorithms have code comments explaining logic
- [ ] Bug fixes documented in commit-style messages (even if no git commit)
- [ ] User-facing features have inline help or tooltips
- [ ] Extensibility points documented (B.4 - plugin architecture notes)

### 8. Backward Compatibility
- [ ] Old YAML configs still load and run
- [ ] Old HDF5 snapshots still readable (with graceful fallback for new attributes)
- [ ] QSettings from previous versions migrate cleanly
- [ ] No breaking API changes to ParticleSystem or Simulation

### 9. Cross-Platform Considerations
- [ ] PyQt6/PyQt5 compatibility handled (if project supports both)
- [ ] File paths use os.path.join (not hardcoded separators)
- [ ] No Windows-specific code (or documented as such)
- [ ] Web app tested in Chrome, Firefox, Edge

### 10. Final Acceptance Checklist
**From AI_PROMPT.md Section 7 (Self-Verification):**
- [ ] Every acceptance criterion has corresponding implementation
- [ ] All placeholder TODOs removed from code
- [ ] All identified bugs have fix commits with clear messages
- [ ] Manual testing checklist fully executed (M.1-M.12)
- [ ] No regressions in existing functionality (old configs still work)
- [ ] Code follows project conventions (snake_case, docstrings, FP32)
- [ ] User can launch both GUIs and complete full workflow without errors

## Acceptance Criteria
**This task is complete when:**
- [ ] All 10 validation steps completed and documented
- [ ] Every acceptance criterion from AI_PROMPT.md is verified (100% coverage)
- [ ] No critical bugs remain (all found bugs are fixed or documented as known issues)
- [ ] Performance meets or exceeds targets
- [ ] System is production-ready (can be shipped to users)
- [ ] Final validation report written (summary of findings)

## Validation Report Format
Create a final report documenting:
```
FINAL VALIDATION REPORT
=======================

1. REQUIREMENT TRACEABILITY
   - Total acceptance criteria: X
   - Implemented: Y
   - Tested: Z
   - Missing/Deferred: [list with justification]

2. SYSTEM INTEGRATION
   - Desktop workflow: PASS/FAIL
   - Web workflow: PASS/FAIL
   - Bug fix verification: PASS/FAIL
   - [Details of any failures]

3. PERFORMANCE MEASUREMENTS
   - Desktop input lag: X ms (target <100ms)
   - Web rendering: Y fps (target 30+ fps)
   - Diagnostics update: Z ms (target <50ms)

4. CODE QUALITY
   - Placeholders remaining: N
   - Test coverage: X%
   - Documentation complete: YES/NO

5. KNOWN ISSUES
   - [List any bugs found but not fixed]
   - [List any deferred features (e.g., volumetric rendering)]

6. PRODUCTION READINESS
   - Ready to ship: YES/NO
   - Blocking issues: [list or "None"]
```

## Code Review Checklist
- [ ] All previous tasks (TASK0-TASK12) are marked complete
- [ ] Every acceptance criterion verified against implementation
- [ ] No skipped or merged requirements without documentation
- [ ] Final report is comprehensive and honest
- [ ] Any deviations from AI_PROMPT.md are justified

## Reasoning Trace
**Why This Task Is Mandatory:**
- Individual tasks can succeed while system fails (integration issues)
- Easy to forget edge cases or requirements during implementation
- Final validation ensures 100% coverage and no information loss
- Production readiness requires system-level thinking, not just component testing

**Validation Philosophy:**
- Trust but verify: previous tasks claim completion, this task verifies
- Traceability: every requirement must have a code location
- Honesty: document what's missing, don't claim false completeness
- Cohesion: system must work as a unified whole

**What Success Looks Like:**
- User can download code and immediately use both GUIs
- All features work as described in AI_PROMPT.md
- No surprises (errors, crashes, missing features)
- Code is maintainable and extensible for future work

**What Failure Looks Like:**
- Missing acceptance criteria (requirements not implemented)
- Integration bugs (components don't work together)
- Poor performance (doesn't meet targets)
- Code quality issues (placeholders, dead code, no docs)

If this task finds issues, create fix tasks or document as known issues. Do NOT mark complete until system is truly production-ready.
