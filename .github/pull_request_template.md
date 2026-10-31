## Summary

<!-- What changed and why. Link the issue if one exists. -->

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format .` pass
- [ ] `uv run mypy src` passes
- [ ] `uv run pytest` passes; new checks have an injected-corruption test on **both** formats (v2.1 + v3.0)
- [ ] README.md **and** README.zh-CN.md updated together (if user-facing)
- [ ] New check tunables are exposed as profile keys
- [ ] Conventional Commit message
