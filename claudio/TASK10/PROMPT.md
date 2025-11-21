## PROMPT
Integrate all desktop GUI components and implement comprehensive error handling for all edge cases. Ensure the full simulation workflow works end-to-end without crashes.

## COMPLEXITY
Medium

## CONTEXT REFERENCE
**Read:** `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

## TASK-SPECIFIC CONTEXT
- **Files:** `gui/main_window.py`, `gui/control_panel.py`, `gui/data_display.py`
- **Error handling:** E.1-E.4 from AI_PROMPT.md
- **Test workflow:** Config → Simulation → Diagnostics → Web Viewer

### Error Handling Pattern
```python
try:
    self.simulation.run()
except Exception as e:
    QMessageBox.critical(self, "Simulation Error", f"Failed to run: {str(e)}")
    logging.exception("Simulation failed")
    self.reset_ui_state()
```

## LAYER
3 (Integration)

## PARALLELIZATION
Parallel with: []

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- All errors must display user-friendly messages
- No crashes allowed (catch all exceptions)
