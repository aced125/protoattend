import protoattend.utils as utils


def test_get_default_configuration():
    default_config = utils.load_default_config()
    assert "candidate_batch_size" in default_config


def test_get_optimizer_names():
    assert "AdaBound" in utils.get_available_optimizers()
