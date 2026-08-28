from olp_repro.distributions import generate_input


def test_shapes():
    inst = generate_input(100, 5, "uniform", seed=0)
    assert inst.r.shape == (100,)
    assert inst.A.shape == (100, 5)
    assert inst.b.shape == (5,)


def test_save_and_load_roundtrip(tmp_path):
    from olp_repro.data_io import dataset_path, load_instance, save_instance

    inst = generate_input(20, 2, "uniform", seed=123)
    path = dataset_path(tmp_path, "uniform", 20, 2, 123)
    save_instance(inst, path, distribution="uniform", seed=123)
    loaded = load_instance(path)

    import numpy as np
    assert np.array_equal(inst.r, loaded.r)
    assert np.array_equal(inst.A, loaded.A)
    assert np.array_equal(inst.b, loaded.b)
