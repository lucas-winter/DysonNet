import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run reduced test matrix for slow/difficult tests.",
    )


def pytest_configure(config):
    if config.getoption("--fast"):
        os.environ.setdefault("PYTEST_FAST", "1")


@pytest.fixture(scope="session")
def fast_mode(pytestconfig):
    return bool(pytestconfig.getoption("--fast"))
