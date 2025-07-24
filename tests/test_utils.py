import tempfile
import yaml
from src.utils import load_config


def test_load_config():
    # Create a temporary config file
    sample_config = {"param1": "value1", "param2": 123}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(sample_config, tmp)
        tmp_path = tmp.name

    # Call the function
    result = load_config(tmp_path)

    # Assert the result matches expected
    assert result == sample_config
