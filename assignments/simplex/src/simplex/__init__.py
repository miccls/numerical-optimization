from jaxtyping import install_import_hook

# Adds jaxtyping decorators, see https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.install_import_hook
with install_import_hook(("simplex", "common"), "beartype.beartype"):
    from simplex import (  # noqa: F401
        dual_simplex,
        linear_algebra,
        pivoting_strategy,
        primal_simplex,
    )
