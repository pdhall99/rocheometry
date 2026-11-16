"""Naming the frozen rounding controls must preserve their binary64 values."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("module", "values"),
    [
        (
            "watershed.traced",
            {
                "WATERSHED_RIM_RADIUS_ROOT_TOLERANCE": 8e-15,
                "WATERSHED_TRIG_CLEANUP_TOLERANCE": 1e-15,
                "WATERSHED_AZIMUTH_MATCH_TOLERANCE": 1e-14,
                "WATERSHED_DEPTH_ROOT_RELATIVE_TOLERANCE": 2e-14,
                "WATERSHED_RIM_STATIONARITY_TOLERANCE": 2e-13,
            },
        ),
        (
            "watershed.family",
            {
                "WATERSHED_FIT_RANK_RELATIVE_TOLERANCE": 1e-10,
                "WATERSHED_FIT_TRIG_CLEANUP_TOLERANCE": 1e-15,
            },
        ),
        ("silhouette.chord", {"LOS_STATIONARY_ROOT_RELATIVE_TOLERANCE": 1e-15}),
        (
            "silhouette.projected",
            {
                "LOS_MAXIMUM_RADIUS_ROOT_RELATIVE_TOLERANCE": 2e-15,
                "LOS_RIM_CROSSING_ROOT_TOLERANCE": 1e-15,
                "LOS_START_ANGLE_MATCH_TOLERANCE": 8e-16,
                "LOS_STATIONARITY_RELATIVE_TOLERANCE": 2e-13,
            },
        ),
        ("silhouette.limb", {"LIMB_JUNCTION_ANGLE_ROOT_TOLERANCE": 8e-16}),
        (
            "silhouette.junction_polish",
            {"LIMB_JUNCTION_NEWTON_STEP_RELATIVE_TOLERANCE": 4e-16},
        ),
        (
            "section_path.axis_contact",
            {
                "AXIS_CONTACT_ROOT_AND_BRACKET_TOLERANCE": 2e-14,
                "AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE": 8e-16,
                "AXIS_CONTACT_FOLD_ROOT_ABSOLUTE_TOLERANCE": 2e-15,
            },
        ),
        (
            "section_path.traced",
            {
                "SECTION_TRIG_CLEANUP_TOLERANCE": 1e-14,
                "SECTION_ARM_ROOT_RELATIVE_TOLERANCE": 8e-16,
                "SECTION_RAY_ROOT_RELATIVE_TOLERANCE": 8e-15,
                "SECTION_ARM_CURVATURE_RESIDUAL_TOLERANCE": 1e-9,
            },
        ),
        ("section_path.contact", {"CONTACT_PLANE_PARALLEL_TOLERANCE": 1e-14}),
    ],
)
def test_rounding_controls_retain_previous_values(
    module: str, values: dict[str, float]
) -> None:
    loaded = import_module("rocheometry._computers." + module)
    for name, value in values.items():
        assert getattr(loaded, name).hex() == value.hex()
