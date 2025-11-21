## PROMPT
Implement or verify the web viewer widget integration in the desktop GUI. Ensure `WebViewerWidget` properly embeds the web app using QWebEngineView and can load simulation snapshots from the output directory.

## COMPLEXITY
Low

## CONTEXT REFERENCE
**Read:** `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

## TASK-SPECIFIC CONTEXT
- **Files:** `gui/web_viewer.py`, `gui/main_window.py`
- **Web app location:** `web/index.html`
- **Use:** QWebEngineView to embed local HTML

### Implementation Pattern
```python
class WebViewerWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.web_view = QWebEngineView()

        # Load local web app
        html_path = os.path.abspath('web/index.html')
        self.web_view.load(QUrl.fromLocalFile(html_path))

        layout.addWidget(self.web_view)
```

## LAYER
1

## PARALLELIZATION
Parallel with: [TASK2, TASK3, TASK4]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Must handle case where web files don't exist
- Test ONLY changed files
