def test_cli_nbwrap_imports():
    import cli_nbwrap as cli

    assert hasattr(cli, "run_all")
