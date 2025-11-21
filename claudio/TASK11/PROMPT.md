## PROMPT
Execute comprehensive manual testing checklist to validate all desktop and web GUI features. Test on Windows 10/11 with ultrawide monitor and RTX 4090. Document results and any issues found.

## COMPLEXITY
Low (execution), Critical (quality assurance)

## CONTEXT REFERENCE
**Read:** `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

## TASK-SPECIFIC CONTEXT
- **Manual tests:** M.1-M.12 from AI_PROMPT.md
- **Configs:** `configs/schwarzschild_tde.yaml`
- **Expected performance:** 30+ fps web, <100ms desktop input lag

### Testing Procedure
1. Launch desktop app
2. Load config
3. Start simulation
4. Verify diagnostics update
5. Test all buttons/controls
6. Open web viewer
7. Load data
8. Test all web controls
9. Export screenshot
10. Document results

## LAYER
3 (Integration/Testing)

## PARALLELIZATION
Parallel with: [TASK12]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Document ALL issues found
- Take screenshots of visual bugs
- Measure actual performance (fps, response times)
