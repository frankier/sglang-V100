# CI cleanup after enabling Actions

The initial main-branch lint run checked the entire V100 fork and found
formatting debt, missing executable bits, and 14 manual tests in the registered
CI tree. The pinned pre-commit formatters normalize the existing sources;
Python AST comparison across all 118 changed Python files confirms executable
statements are unchanged, except for removing five unused imports. The checks
also ignore docstring formatting and import ordering. The result is retained
in [python-format-check.json](python-format-check.json).

Unified-diff `.patch` files retain their original payload bytes. The trailing
whitespace and EOF hooks now exclude that file format because its whitespace
is meaningful. Historical benchmark outputs and tested-source hashes remain
records of the original runs; their JSON values are unchanged by adding final
newlines. The PR-status helper treats an unindexed workflow's HTTP 404 as
"not run", while preserving other API errors.

The manual tests move in a separate, byte-identical commit:
`193c636ada7d42f81e353f03f92bee0ac268410b`. They remain runnable under
`test/manual/`, with current reproduction commands updated accordingly. No
registered CI test is disabled to make lint pass.

## Reproduce the manual-test move

The [proof archive](manual-test-move-proof.zip) contains the standalone
reproduction script, its helper, and the successful output. From this
repository's root, with Python, Git, and pre-commit installed:

```bash
mkdir -p "$HOME/.scratch/sglang-v100-move-proof"
unzip benchmark/v100_ci_20261111/manual-test-move-proof.zip -d "$HOME/.scratch/sglang-v100-move-proof"
TMPDIR="$HOME/.scratch" python3 "$HOME/.scratch/sglang-v100-move-proof/reproduce.py"
```

The script creates an isolated scratch worktree at the parent commit, repeats
the 14 file moves, runs the relevant pre-commit hooks, and requires an empty
diff against the move commit. It prints the cleanup command for its worktree.
