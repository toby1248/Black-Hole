## PROMPT
**FINAL SYSTEM VALIDATION - MANDATORY**

Perform comprehensive system-level validation to ensure the TDE-SPH GUI system is complete, coherent, and production-ready. Verify EVERY acceptance criterion from AI_PROMPT.md is implemented and tested. This is the cohesion validation layer that confirms the system works as a unified whole.

## COMPLEXITY
Medium (verification and documentation work)

## CONTEXT REFERENCE
**For complete environment context, read:**
- `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md` - Contains ALL acceptance criteria to verify

**You MUST read AI_PROMPT.md before executing this task to understand what to validate.**

## TASK-SPECIFIC CONTEXT

### Validation Scope
This task verifies implementation of ALL acceptance criteria from AI_PROMPT.md:
- **Section 1:** Desktop Diagnostics (C.1-C.4, A.1)
- **Section 2:** Desktop Buttons (B.1-B.4)
- **Section 3:** Web Volumetric Rendering (V.1-V.4, optional)
- **Section 4:** Web Buttons (D.1-D.17)
- **Section 5:** Bug Fixes (T.1-T.7)
- **Section 6:** Edge Cases (E.1-E.8)
- **Section 7:** Testing (M.1-M.12, A.1-A.4)

### Validation Methodology
**For each acceptance criterion:**
1. Find implementation in codebase (file:line reference)
2. Verify test coverage exists (unit test or manual test result)
3. Test the feature yourself (execute and verify behavior)
4. Document verification status (✓ implemented, ✗ missing, ⚠ partial)

**Example:**
```
Criterion C.1: Diagnostics tab displays comprehensive particle statistics
✓ Implementation: gui/data_display.py:350-450 (DiagnosticsWidget)
✓ Test: gui/data_display.test.py:20-45 (test_diagnostics_update)
✓ Manual verification: Launched GUI, started sim, verified all stats displayed
Status: COMPLETE
```

### Integration Tests
Execute these end-to-end workflows and verify all steps work:

**Desktop Workflow:**
1. Launch `python gui/main_window.py`
2. Load `configs/schwarzschild_tde.yaml`
3. Click Start Simulation
4. Verify diagnostics panel updates in real-time
5. Toggle unit conversion (code ↔ physical units)
6. Open Preferences, change setting, verify persistence
7. Open web viewer tab, verify particles render
8. Stop simulation
Expected: No errors, all data displays correctly

**Web Workflow:**
1. Open `web/index.html` in Chrome
2. Click "Load Demo Data"
3. Verify particles render in 3D view
4. Click Play animation
5. Change colormap to "plasma"
6. Adjust point size slider
7. Click "Export Screenshot"
Expected: Smooth playback, colors update, screenshot downloads

**Bug Fix Verification:**
1. Start simulation
2. Check temperature data in diagnostics (should be >0, not 0.0 fallback)
3. Open HDF5 snapshot with h5py
4. Verify temperature array exists and has values
Expected: Temperature bug is fixed

### Performance Validation
Measure actual performance:
- Desktop: Click button, measure response time (should be <100ms)
- Web: Enable FPS counter, verify 30+ fps with demo data
- Diagnostics: Time update from signal to display (should be <50ms)

### Code Quality Checks
Search for quality issues:
```bash
# Find placeholder code
grep -r "TODO\|FIXME\|NotImplemented" gui/ web/js/

# Find defensive patterns (should be removed)
grep -r "hasattr.*temperature\|getattr.*temperature" src/ gui/

# Find debug prints (should be removed)
grep -r "print(" gui/ src/ | grep -v "# debug"
```

### Final Report Template
Document findings in this format:
```
FINAL VALIDATION REPORT
=======================
Date: [YYYY-MM-DD]
Validator: [Agent ID or name]

1. REQUIREMENT TRACEABILITY
   Total criteria: 80+ (estimated from AI_PROMPT.md)
   Implemented: X
   Tested: Y
   Missing: Z
   Deferred: [list with justification]

2. SYSTEM INTEGRATION TESTS
   ✓/✗ Desktop workflow
   ✓/✗ Web workflow
   ✓/✗ Bug fix verification
   ✓/✗ Unit conversion
   ✓/✗ Preferences persistence

3. PERFORMANCE MEASUREMENTS
   Desktop input lag: X ms (target <100ms)
   Web rendering: Y fps (target 30+ fps)
   Diagnostics update: Z ms (target <50ms)
   PASS/FAIL

4. CODE QUALITY
   Placeholders found: N
   Defensive patterns found: M
   Debug prints found: K
   Test coverage: X%
   Documentation: Complete/Incomplete

5. KNOWN ISSUES
   [List bugs found but not fixed]
   [List deferred features]

6. PRODUCTION READINESS: YES/NO
   Blocking issues: [None or list]

CONCLUSION:
[System is/is not ready for production because...]
```

## EXTRA DOCUMENTATION

### What to Do If Issues Found
**If missing implementation:**
- Document as "Known Issue: Feature X not implemented"
- Justify why (e.g., "Volumetric rendering deferred due to complexity")

**If bugs found:**
- Fix immediately if simple (< 1 hour)
- Document as "Known Issue" if complex
- Do NOT mark validation complete if critical bugs exist

**If tests missing:**
- Write minimal tests to verify functionality
- Or document as "Known Issue: Feature X not tested"

### Success Criteria
This task is ONLY complete when:
- [ ] Every acceptance criterion verified (100% coverage)
- [ ] All critical features work (desktop diagnostics, web visualization, bug fixes)
- [ ] No critical bugs remain
- [ ] Performance meets targets
- [ ] Final report written and comprehensive

Do NOT mark this task complete if system is not production-ready.

## LAYER
Ω (Final System Validation)

## PARALLELIZATION
Parallel with: []

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- MUST verify EVERY acceptance criterion (no skipping)
- MUST test end-to-end workflows (not just unit tests)
- MUST measure actual performance (not assume)
- MUST be honest about missing features or bugs
- Final report MUST be comprehensive
- If system not ready, document why and what's missing
