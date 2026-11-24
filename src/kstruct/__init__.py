"""KStruct: mechanical part FEA for non-experts."""

RELEASE_VERSION = "0.3.2"  # keep in sync with pyproject.toml (tests check)

# The build number is stamped into kstruct/_build.py by packaging/build.py (and by the CI
# workflow) right before freezing, so every build gets a higher version:
# RELEASE_VERSION.BUILD_NUMBER. A source checkout without the stamp runs as the plain
# release version.
try:
    from kstruct._build import BUILD_DATE, BUILD_NUMBER, GIT_COMMIT
except ImportError:  # not stamped (development checkout)
    BUILD_NUMBER, BUILD_DATE, GIT_COMMIT = 0, "", ""

__version__ = f"{RELEASE_VERSION}.{BUILD_NUMBER}" if BUILD_NUMBER else RELEASE_VERSION


def build_info() -> str:
    """'0.1.0.57 (2026-10-27, 89848bb)' or just the version for an unstamped checkout."""
    details = ", ".join(x for x in (BUILD_DATE, GIT_COMMIT) if x)
    return f"{__version__} ({details})" if details else __version__
