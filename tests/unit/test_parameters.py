from collections import OrderedDict

import torch

from dream.parameters import ParameterSpec


def test_flatten_round_trip() -> None:
    params = OrderedDict(
        weight=torch.arange(6, dtype=torch.float64).view(2, 3),
        bias=torch.tensor([-1.0, 2.0], dtype=torch.float64),
    )
    spec = ParameterSpec.from_params(params)
    restored = spec.unflatten(spec.flatten(params))
    assert tuple(restored) == tuple(params)
    for name in params:
        torch.testing.assert_close(restored[name], params[name])

