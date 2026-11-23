
import yaml

from src.data import load_dataset


def loadConfig():
    with open("config/train.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def testConfigContainsRequiredSections():
    cfg = loadConfig()

    for section in ["data", "split", "model", "artifacts_dir"]:
        assert section in cfg, f"Missing section: {section}"


def testTestSizeIsValid():
    cfg = loadConfig()

    testSize = float(cfg["split"]["test_size"])

    assert 0 < testSize < 1


def testDatasetIsNotEmpty():
    cfg = loadConfig()

    X, y = load_dataset(cfg)

    assert len(X) == len(y) > 0
