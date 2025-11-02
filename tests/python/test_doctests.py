import doctest

import busy_beaver_blaze.interactive as interactive_module


def test_doctests_interactive() -> None:
    """Ensure doctest snippets in interactive module stay valid."""
    failures, _ = doctest.testmod(
        interactive_module,
        optionflags=doctest.NORMALIZE_WHITESPACE,
    )
    assert failures == 0, f"Doctest failures detected in {interactive_module.__name__}"
