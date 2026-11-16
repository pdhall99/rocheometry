"""Pickled geometry retains configuration and process-local graph sharing."""

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import cast

import pytest

import rocheometry
from rocheometry import Fitted
from rocheometry._computers.composition import Memoized


@dataclass
class _CountedSum:
    calls: int = 0

    def __call__(self, value: float, *, offset: float = 0.0) -> float:
        self.calls += 1
        return value + offset


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("maxsize", [0, 2, None])
def test_memoized_pickle_preserves_configuration_and_starts_empty(
    protocol: int, maxsize: int | None
) -> None:
    computer = _CountedSum()
    original = Memoized(computer, maxsize=maxsize)
    assert original(1.0, offset=2.0) == 3.0
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))

    assert restored is not original
    assert restored.maxsize == maxsize
    assert restored.cache_info() == (0, 0, maxsize, 0)
    assert restored.computer is not computer
    assert restored.computer.calls == 1
    assert restored(1.0, offset=2.0) == 3.0
    assert restored(1.0, offset=2.0) == 3.0
    assert restored.computer.calls == (3 if maxsize == 0 else 2)
    assert computer.calls == 1
    restored.cache_clear()
    assert restored.cache_info() == (0, 0, maxsize, 0)


def test_separate_default_geometry_pickles_reuse_the_process_graph() -> None:
    component = rocheometry.ComponentGeometry(0.3)
    binary = rocheometry.BinaryGeometry(0.3, separation=2.0, frame="centre_of_mass")
    component.l1()

    restored_component = pickle.loads(pickle.dumps(component))
    restored_binary = pickle.loads(pickle.dumps(binary))

    assert restored_component == component
    assert restored_component.field == component.field
    assert restored_component._computers is Fitted()._computers()
    assert restored_component.l1() is component.l1()
    assert restored_binary == binary
    assert restored_binary._computers is restored_component._computers
    assert restored_binary.geometry1._computers is restored_binary._computers
    assert restored_binary.geometry2._computers is restored_binary._computers
    assert restored_binary.component(2).intrinsic_geometry is restored_binary.geometry2


def test_selected_surfaces_and_views_can_be_pickled() -> None:
    component = rocheometry.ComponentGeometry(0.3)
    surface = component.surface(volume_radius_fill=0.8)
    view = (
        rocheometry.BinaryGeometry(0.3, separation=2.0)
        .component(2)
        .surface(volume_radius_fill=0.8)
    )

    restored_surface = pickle.loads(pickle.dumps(surface))
    restored_view = pickle.loads(pickle.dumps(view))

    assert restored_surface == surface
    assert restored_view == view
    assert restored_view.intrinsic_geometry.component is (
        restored_view.component.intrinsic_geometry
    )
    assert restored_surface.component._computers is Fitted()._computers()
    assert restored_view.component.binary._computers is Fitted()._computers()


@pytest.mark.parametrize(
    "accuracy",
    (
        rocheometry.Fitted(),
        rocheometry.Reference(),
        rocheometry.Reference(polar_order=16, azimuth_order=8),
    ),
)
@pytest.mark.parametrize(
    "kind", ("component", "surface", "binary", "view", "view_surface")
)
def test_public_values_pickle_only_settings_and_reconnect(
    accuracy: rocheometry.Fitted | rocheometry.Reference, kind: str
) -> None:
    component = rocheometry.ComponentGeometry(0.3, accuracy=accuracy)
    binary = rocheometry.BinaryGeometry(
        0.3, separation=12, frame="centre_of_mass", accuracy=accuracy
    )
    values = {
        "component": component,
        "surface": component.lobe(),
        "binary": binary,
        "view": binary.component(2),
        "view_surface": binary.component(2).lobe(),
    }
    value = values[kind]
    payload = pickle.dumps(value)
    assert len(payload) < 1000
    assert b"ComputerSet" not in payload
    restored = pickle.loads(payload)
    assert restored == value
    if isinstance(restored, rocheometry.ComponentGeometry):
        graph = restored._computers
    elif isinstance(restored, rocheometry.ComponentSurfaceGeometry):
        graph = restored.component._computers
    elif isinstance(restored, rocheometry.BinaryGeometry):
        graph = restored._computers
        assert graph is restored.geometry1._computers is restored.geometry2._computers
    elif isinstance(restored, rocheometry.BinaryComponentView):
        graph = restored.binary._computers
    else:
        graph = restored.component.binary._computers
        assert (
            restored.intrinsic_geometry.component
            is restored.component.intrinsic_geometry
        )
    assert graph is accuracy._computers()


def _worker_graph(geometry: rocheometry.ComponentGeometry) -> tuple[bool, int, int]:
    node = cast(Memoized[..., rocheometry.CriticalPoint], geometry._computers.l1)
    before = node.cache_info()
    geometry.l1()
    after = node.cache_info()
    return (
        geometry._computers is geometry.accuracy._computers(),
        after.hits - before.hits,
        after.misses - before.misses,
    )


@pytest.mark.parametrize(
    "accuracy",
    (rocheometry.Fitted(), rocheometry.Reference(polar_order=16, azimuth_order=8)),
)
def test_spawned_worker_reuses_settings_graph_across_separate_tasks(
    accuracy: rocheometry.Fitted | rocheometry.Reference,
) -> None:
    geometry = rocheometry.ComponentGeometry(0.3, accuracy=accuracy)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        assert pool.submit(_worker_graph, geometry).result(timeout=30) == (True, 0, 1)
        assert pool.submit(_worker_graph, geometry).result(timeout=30) == (True, 1, 0)
