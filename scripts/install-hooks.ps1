Set-Location (Split-Path -Parent $PSScriptRoot); git config core.hooksPath .githooks   # once; worktrees share the repo config, so it covers every session
