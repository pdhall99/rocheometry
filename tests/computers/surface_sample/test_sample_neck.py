"""Thin contact nodes retain their root displacement through local geometry."""

import pickle

from rocheometry._computers.surface_sample.sample import _NeckSample


def test_retained_node_gradient_survives_pickling() -> None:
    direction = (1.0, 0.0, 0.0)
    gradient = (-1.0, 0.0, 0.0)
    sample = _NeckSample(
        ((direction, 1.0),),
        (0.5,),
        True,
        origin=0.5,
        depth=1e-20,
        _gradients={direction: gradient},
    )
    restored = pickle.loads(pickle.dumps(sample))
    assert restored == sample
    assert restored._gradients[direction] == gradient
