from jaxtyping import install_import_hook

# Adds jaxtyping decorators, see https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.install_import_hook
with install_import_hook("simplex", "beartype.beartype"):
    from simplex import lp_problem, pivoting_strategy, solver  # noqa: F401
