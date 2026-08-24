import torch.nn as nn

from dream.models import ResNet20GN1


def test_reference_resnet_is_stateless_and_matches_parameter_count() -> None:
    model = ResNet20GN1()
    assert sum(parameter.numel() for parameter in model.parameters()) == 278132
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())

