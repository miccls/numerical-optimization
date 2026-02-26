from jaxtyping import install_import_hook

# Adds jaxtyping decorators, see https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.install_import_hook
with install_import_hook(("simplex", "common"), "beartype.beartype"):
    from common import lp_problem  # noqa: F401

    from simplex import pivoting_strategy, solver  # noqa: F401
