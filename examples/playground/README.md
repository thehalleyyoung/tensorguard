# TensorGuard Local Playground

Generate a self-contained, no-upload playground:

```bash
tensorguard playground --output tensorguard_playground
open tensorguard_playground/index.html
```

The page is static and private by construction: examples are read as text,
verified through the AST-only safe loader, and embedded with deterministic
verdict snapshots. Edit the examples in the page, then copy a variant into a
local file and run `tensorguard verify path.py` to re-analyze it.
