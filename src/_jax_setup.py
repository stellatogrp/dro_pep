"""JAX one-time setup. Import this BEFORE any module that uses JAX.

Enables the persistent compilation cache so re-running an experiment in a
fresh process skips the ~50 s HLO+XLA recompile cycle. The cache is
worktree-local by default (``<repo_root>/.jax_compilation_cache/``) so
parallel sessions in sibling worktrees don't fight over a shared directory.

Override the location by exporting ``JAX_COMPILATION_CACHE_DIR=...`` before
launch — this module respects it.
"""
import os

import jax

if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    _here = os.path.dirname(os.path.abspath(__file__))
    _default_dir = os.path.abspath(os.path.join(_here, "..", ".jax_compilation_cache"))
    os.makedirs(_default_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _default_dir)

# Defaults skip caching small/fast compiles; for this codebase even the
# smaller pjits add up, so cache everything.
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
