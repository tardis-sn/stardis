def test_stardis_output(example_stardis_output, example_tracing_nus):
    assert len(example_stardis_output.spectrum_nu) == len(example_tracing_nus)
