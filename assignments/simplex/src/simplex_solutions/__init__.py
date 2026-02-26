from jaxtyping import install_import_hook

# Adds jaxtyping decorators, see https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.install_import_hook
with install_import_hook(("simplex_solutions", "common"), "beartype.beartype"):
    from common import lp_problem
    from simplex_solutions import (  # noqa: F401
        linear_algebra,
        pivoting_strategy,
        solver,
    )
