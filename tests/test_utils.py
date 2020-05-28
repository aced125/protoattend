import protoattend.utils as utils


def test_get_default_configuration():
    default_config = utils.load_default_config()
    assert "candidate_batch_size" in default_config
