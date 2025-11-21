@dependencies [TASK0]
# Task: Implement Web Viewer Widget Integration

## Summary
Ensure the web viewer widget (`WebViewerWidget`) properly embeds the web app and can load simulation snapshots. This connects the desktop GUI to the web-based 3D visualization.

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **Files:** `gui/web_viewer.py`, `gui/main_window.py`
- **Pattern:** Use QWebEngineView to embed web app (likely already partially implemented)
- **Integration:** Load local `web/index.html` or connect to Flask server

## Complexity
Low

## Dependencies
Depends on: [TASK0]
Blocks: [TASK11]
Parallel with: [TASK2, TASK3, TASK4]

## Detailed Steps
1. Verify `WebViewerWidget` uses `QWebEngineView`
2. Load `web/index.html` on widget initialization
3. Add refresh button to reload current snapshot
4. Implement data passing from desktop to web (via URL parameters or JavaScript bridge)
5. Test embedded viewer displays correctly

## Acceptance Criteria
- [ ] **B.3** Web viewer widget functional
- [ ] Embeds web app via QWebEngineView
- [ ] Can load snapshots from output directory
- [ ] Refresh button reloads current snapshot
- [ ] Integration tested with desktop GUI

## Code Review Checklist
- [ ] QWebEngineView properly initialized
- [ ] Error handling for missing web files
- [ ] JavaScript console errors logged
- [ ] Follows PyQt conventions
