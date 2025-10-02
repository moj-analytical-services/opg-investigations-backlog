def test_imports():
    import src.opg_backlog_sim.emulator as emu
    import src.opg_backlog_sim.bo as bo
    import src.opg_backlog_sim.causal_survival as cs
    assert hasattr(emu,'train_emulator')
