@dependencies [TASK0, TASK1, TASK2, TASK3, TASK4, TASK5]
# Task: Desktop GUI Integration and Error Handling

## Summary
Integrate all desktop GUI components (diagnostics, unit conversion, preferences, web viewer) and implement comprehensive error handling for edge cases (simulation failures, invalid configs, missing data).

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **Files to modify:** `gui/main_window.py`, `gui/control_panel.py`, `gui/data_display.py`
- **Error scenarios:** E.1-E.4 from AI_PROMPT.md
- **Integration:** Connect all widgets, ensure signals/slots work, test full workflow

## Complexity
Medium

## Dependencies
Depends on: [TASK0, TASK1, TASK2, TASK3, TASK4, TASK5]
Blocks: [TASK11, TASKW]
Parallel with: []

## Acceptance Criteria
- [ ] **E.1** Simulation failure: displays error message, logs traceback, resets UI
- [ ] **E.2** Invalid YAML: shows error dialog, highlights syntax error
- [ ] **E.3** No snapshots: diagnostics show "No data", web viewer shows message
- [ ] **E.4** GPU not available: shows warning, continues with CPU, updates performance panel
- [ ] All desktop components integrated and connected
- [ ] Full workflow tested: load config → start sim → view diagnostics → open web viewer
- [ ] No crashes or unhandled exceptions

## Code Review Checklist
- [ ] Try/except blocks around all simulation operations
- [ ] QMessageBox used for user-facing errors
- [ ] Console logging for debug/traceback
- [ ] UI state correctly reset after errors
- [ ] Thread-safe signal/slot connections
