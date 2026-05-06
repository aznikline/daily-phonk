# AGENTS.md

## Scope
- This file governs work inside `/Users/wizout/op/daily-phonk`.
- The repo is for daily music generation, prompt/config tuning, and lightweight automation around outputs.

## Structure
- `src/` contains the generator implementation.
- `outputs/` is generated audio output and should stay treated as runtime artifacts.
- `requirements.txt` and local config govern environment/runtime behavior.

## Workflow
- Prefer minimal changes to prompt/config/backend selection before rewriting generation code.
- When changing generation behavior, identify whether the change is config-only, DSP-path, or neural-path.
- Keep automation and generation concerns separate.

## Quality bar
- Use the shortest relevant smoke path first.
- If code changes affect the main generator path, validate with the narrowest local run available, typically `python -m src.main`.
- Report whether the check was config-level only or full generation.

## Safety
- Do not commit large generated audio artifacts unless explicitly requested.
- Call out model download, disk, or runtime-cost implications when changing the neural backend path.
- Avoid accidental output sprawl at repo root; generated files belong under `outputs/`.

## Output
- State whether the change affects prompts/config, DSP generation, neural generation, or automation.
- Include files changed, validation run, and any runtime cost or artifact impacts.
