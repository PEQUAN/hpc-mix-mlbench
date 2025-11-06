def pytest_addoption(parser):
    parser.addoption("--alias",
        action="append",
        default=[],
        help="some alias to pass to test functions",)
    parser.addoption("--verbosityLog",
        action="append",
        default=[],
        help="some alias to pass to test functions",)
