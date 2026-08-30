import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import get_model


def test_model_builds():
    model = get_model("resnet18", 10)
    assert model is not None


def test_output_shape():
    model = get_model("resnet18", 10)
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 10)


def test_num_classes_is_configurable():
    model = get_model("resnet18", 5)
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 5)


def test_bad_architecture_raises():
    try:
        get_model("vgg16", 10)
        raised = False
    except ValueError:
        raised = True

    assert raised
