def test_imports():
    import src.opg_backlog_sim.emulator as emu

    assert hasattr(emu, "train_emulator")
