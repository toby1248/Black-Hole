## Plan: agent instructions

The goal is to extend, not rewrite: add new classes and configs, tighten interfaces, and ensure IO/visualization can distinguish “mode” and units. Agents should edit only where needed, primarily per-submodule `CLAUDE.md` plus select code hotspots. Contextless reviewer agents will double-check the logic of each self contained section without bias and distraction 

### Steps
1. Review `CLAUDE.md` and code in each subpackage (`core`, `sph`, `gravity`, `metric`, `integration`, `ICs`, `eos`, `radiation`, `io`, `visualization`, `config`) to align docs with actual Phase 2 state and Phase 3 goals. If there is no CLAUDE.md in any subpackage's top level folder create one.
2. Create a NOTES.md in each folder that contains a CLAUDE.md and instruct the sub agents to use it.
5. Identify and document major architectural or numerical risks in \subpackage\`CLAUDE.md` where appropriate (e.g., timestep control in strong fields, energy diagnostics in GR, unit conventions), so implementation agents can design tests and validation strategies.
6. In each subfolder containing an agent's task add a short “Phase 3 tasks for agents” section to `CLAUDE.md` in that folder, listing concrete responsibilities and cross-module expectations (e.g., `core` config flags, GR metric hooks). Ensure the coder agents also have access to the global CLAUDE.md instructions context
7. Once the coder agent has finished these sections modify \subpackage\`CLAUDE.md` again and add to it the reviewer sub-agent prompt found at the bottom of this file, and spawn the reviewer with no context outside the atomic task's directory

### Further Considerations
	- If goals have changed or modules have been added update the `CLAUDE.md` files with fresh information
	- Encourage agents to maintain full backward compatibility: Newtonian examples/tests must still pass, so new GR paths should be opt-in via config and cleanly tested.
	- FP32 by default, FP64 only where necessary, precision agnostic preferred
	- Instruct agents to consider compatibility with GPU mixed precision tensor computation with linear algebra to be a major benefit for current or future use
	- Don't use absolute filepaths. Export raw data to the output folder and all visualisations and summary documents to results folder
	- Catch and mitigate errors from extreme interactions. Identify and flag bad data and throw warnings for debug. Filter visualisation scaling for extreme outliers



## Reviewer Sub-Agent Prompt
ROLE: Unbiased AI-to-AI code reviewer agent. Succinct and blunt.
SCOPE: Review code in the directory containing this file. To you nothing else exists.
GOAL: Verify the high level functionality and logic of the module

TASKS:
1. Identify correctness and robustness issues.
2. Flag missing context or hidden assumptions.
3. Suggest minimal, concrete improvements.
4. Add to or create the local NOTES.md




