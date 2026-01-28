def test_cli_modules_import():
    # Ensure wrappers import cleanly (indirectly checks notebook_code presence)
    import preprocessing as p
    import intervals as itv
    import analysis_demo as demo
    import distributions as dist

    assert hasattr(p, "engineer")
    assert hasattr(itv, "build_backlog_series")
    assert hasattr(demo, "last_year_by_team")
    assert hasattr(dist, "interval_change_distribution")
