"""Published Roche tables against Rocheometry.

Each table is an ECSV file in ``tables/``, transcribed as
printed (see ``tables/README.md`` for how a table is added and how a
disagreement is explained). Its conventions say how a row becomes a
Rocheometry geometry, and each column names the quantity it holds. One job
compares one table at Reference accuracy.

A value's error is ``d``, the difference from the printed value in
half-units of its last printed digit, and its bound is the largest of one
half-unit (rounding), the table's or column's allowance, and the
``max_half_units`` of a note explaining that value. Tables still in triage
give note rows, so they never fail.

Columns that are not Rocheometry outputs are not compared here: their status
in the table says why, or their quantity is in ``NOT_OUTPUTS``, or they are
the vertex's x and z, or a crossing of an L2 or L3 level.
"""

from __future__ import annotations

import csv
import math
from decimal import Decimal
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

import yaml

import rocheometry
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study
from rocheometry._component_geometry import RadiusName

TABLES = Path(__file__).parent / "tables"

METRIC = "half-units of the last printed digit"

#: Tabulated quantities Rocheometry does not report, with the reason.
NOT_OUTPUTS = {
    "radial_area_integral": "Leahy & Leahy's 'area' is the integral of r^2 over solid angle, not the area",
    "own_mass_fraction": "a convention, not a computed quantity",
}


class LiteratureTables(Study):
    name = "literature_tables"
    kind = "literature"
    quantities = (
        "critical.l1",
        "lobe.max_fill",
        "critical.outer",
        "lobe.volume_radius",
        "integrals.volume",
        "integrals.area",
        "integrals.mean_gravity",
        "radii.named",
        "radii.direction",
        "outlines.section",
        "selection.by_fill_out_factor",
        "potential.conventions",
    )
    domain = (
        "Kuiper & Johnson (1956), Kopal (1959), Plavec & Kratochvil (1964), "
        "Mochnacki (1984), Leahy & Leahy (2015) and Guillochon & Prochaska (2026): "
        "L1, L2 and L3, lobe radii and volumes, the radii, volumes, areas and mean "
        "gravities of detached surfaces, and the potentials, volumes, areas and mean "
        "gravities of contact envelopes, for mass ratios from 0.01 to 100, and the "
        "lobe volume radius and L1 from 1e-6 to 1e6."
    )

    def jobs(self) -> list[dict[str, str]]:
        """One job per table with a compared column; the others are only read as partners."""
        return [
            dict(table=path.stem)
            for path in tables()
            if any(compared(f) for f in load(path.stem)["fields"])
        ]

    def smoke(self) -> dict[str, str]:
        return dict(table="kuiper_johnson1956_table1")

    def measure(self, job: dict[str, str]) -> list[Row]:
        table = load(job["table"])
        meta = table["meta"]
        conventions = meta["conventions"]
        mass = conventions["mass_parameter"]
        digits = meta.get("significant_digits")
        source = meta["reference"]
        triage = meta.get("status") == "triage"
        fields = [f for f in table["fields"] if compared(f)]
        rows = []
        # Rows outside Rocheometry's domain, such as a printed q = 0 limit.
        skipped = meta.get("row_status", {})
        notes = meta.get("notes", {})
        for row in table["rows"]:
            if row[mass["column"]] in skipped:
                continue
            for field in fields:
                qi, m = component(row[mass["column"]], mass, field)
                if (
                    field["rocheometry_quantity"] in ENVELOPE
                ):  # a total of printed values
                    key, actual, expected, half = envelope(
                        field, geometry(qi), conventions, row
                    )
                    shown: dict[str, Any] = dict(printed_total=expected)
                else:
                    key, actual = evaluate(field, geometry(qi), m, conventions, row)
                    text = row[field["name"]]
                    expected, half = float(text), half_unit(text, digits)
                    shown = dict(printed=text)
                d = (actual - expected) / half
                allowance = field.get(
                    "allowance_half_units", meta.get("allowance_half_units", 1.0)
                )
                if "allowance_relative" in field:
                    allowance = max(
                        allowance, field["allowance_relative"] * abs(expected) / half
                    )
                note = notes.get(row[mass["column"]], {}).get(field["name"])
                bound = max(1.0, allowance, note["max_half_units"] if note else 0.0)
                rows.append(
                    record(
                        key,
                        "reference",
                        abs(d),
                        bound,
                        dict(
                            qi=qi,
                            source=source,
                            table=job["table"],
                            row=row[mass["column"]],
                            column=field["name"],
                        ),
                        metric=METRIC,
                        note="table in triage" if triage else None,
                        verdict=verdict(d, allowance, note),
                        **shown,
                        actual=actual,
                        d=d,
                        explanation=note["class"] if note else None,
                    )
                )
        return rows


def tables() -> list[Path]:
    return sorted(TABLES.glob("*.ecsv"))


@cache
def load(name: str) -> dict[str, Any]:
    """Load a table from its ECSV file.

    Returns:
        The table's ``meta``, its ``fields`` (each column's name, description
        and meta) and its ``rows`` (dicts of the printed strings).
    """
    lines = (TABLES / f"{name}.ecsv").read_text(encoding="utf-8").splitlines()
    header = [line[2:] for line in lines if line.startswith("#")]
    if header[:2] != ["%ECSV 1.0", "---"]:
        raise ValueError(f"{name}: not an ECSV 1.0 file")
    spec = yaml.safe_load("\n".join(header[2:]))
    fields = [
        {k: v for k, v in column.items() if k not in ("datatype", "meta")}
        | column.get("meta", {})
        for column in spec["datatype"]
    ]
    reader = csv.reader(  # runs of spaces are allowed, so columns may be aligned
        (line for line in lines if not line.startswith("#")),
        delimiter=" ",
        skipinitialspace=True,
    )
    names = next(reader)
    if names != [f["name"] for f in fields]:
        raise ValueError(f"{name}: column names differ from the header")
    rows = [dict(zip(names, values, strict=True)) for values in reader]
    return dict(meta=spec["meta"], fields=fields, rows=rows)


def compared(field: dict[str, Any]) -> bool:
    quantity = field.get("rocheometry_quantity")
    if quantity is None or "status" in field or quantity in NOT_OUTPUTS:
        return False
    if quantity == "axis_crossing":  # levels through L2 or L3 enclose both stars
        return field["level"]["point"] == "L1"
    if quantity == "lobe_vertex":  # the section's extent gives y, but not x or z
        return field["coordinate"] == "y"
    return True


# --- conventions: a row and column to a Rocheometry geometry, and back -------------


def component(
    value: str, mass: dict[str, Any], field: dict[str, Any]
) -> tuple[float, float]:
    """(qi, own mass fraction) of the component a column describes."""
    x = float(Fraction(value))
    if mass.get("log10"):  # the column holds log10 of the mass parameter
        x = 10.0**x
    kind = mass["kind"]
    if kind == "secondary_to_primary_ratio":  # each column names its component
        if field.get("component", "secondary") == "primary":
            return 1.0 / x, 1.0 / (1.0 + x)
        return x, x / (1.0 + x)
    if kind == "own_mass_fraction":
        return x / (1.0 - x), x
    if kind == "own_mass_ratio":
        return x, x / (1.0 + x)
    if kind == "companion_mass_ratio":
        return 1.0 / x, 1.0 / (1.0 + x)
    raise ValueError(f"unknown mass parameter {kind!r}")


def frame_x(x: float, m: float, frame: dict[str, Any] | None) -> float:
    """Convert a component-local x to the table's frame.

    Component-local x puts the component at 0 and the companion at +1.
    """
    if not frame or frame["kind"] == "component_local":
        return x
    kind = frame["kind"]
    if kind == "centre_of_mass":
        return (1.0 - m) - x if frame["component_on"] == "+x" else x - (1.0 - m)
    if kind == "companion_origin":
        return 1.0 - x
    raise ValueError(f"unsupported frame {frame!r}")


def potential_spec(
    field: dict[str, Any], conventions: dict[str, Any]
) -> dict[str, Any]:
    spec = field.get("potential")
    return spec if isinstance(spec, dict) else conventions["potential"]


def in_convention(u: float, qi: float, spec: dict[str, Any]) -> float:
    """Rocheometry's Jacobi U in the column's potential convention."""
    kind = spec["kind"]
    if kind == "jacobi_centre_of_mass":  # Mochnacki's C = 2U has scale 2
        return float(spec.get("scale", 1)) * u
    if kind == "plavec_kratochvil_omega":  # Omega = (1 + m) U, m = M2/M1 <= 1
        return (1.0 + min(qi, 1.0 / qi)) * u
    if kind in ("kopal_omega", "kopal_r0"):  # about this component: q = qi
        omega = rocheometry.convert_potential(
            rocheometry.Potential.jacobi(u),
            q=qi,
            convention=rocheometry.PotentialConvention.KOPAL,
            component_id=1,
        ).value
        return omega if kind == "kopal_omega" else 1.0 / (omega - 1.0 / qi)
    raise ValueError(f"unsupported potential {spec!r}")


def scale(conventions: dict[str, Any]) -> float:
    return float(conventions["potential"].get("scale", 1))


def level(
    field: dict[str, Any], g: rocheometry.ComponentGeometry, row: dict[str, str]
) -> float:
    """Jacobi U of a detached surface given by its fill-out F = U_L1 / U."""
    spec = field["level"]
    if spec["kind"] != "fill_out_detached":
        raise ValueError(f"unsupported level {spec!r}")
    fill = (
        row[spec["fill_out_column"]] if "fill_out_column" in spec else spec["fill_out"]
    )
    return g.l1().potential.value / float(fill)


@cache
def geometry(qi: float) -> rocheometry.ComponentGeometry:
    return rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())


@cache
def surface(qi: float, u: float) -> rocheometry.ComponentSurfaceGeometry:
    return geometry(qi).surface(potential=rocheometry.Potential.jacobi(u))


def lagrange(name: str, g: rocheometry.ComponentGeometry) -> tuple[float, float]:
    """(local x, Jacobi U) of L2 (behind the lighter star) or L3 (behind the heavier).

    At equal masses the companion's outer point is L2, as Leahy & Leahy have it.
    """
    if (g.qi < 1.0) == (name == "L2"):
        point = g.outer()
        return point.position[0], point.potential.value
    point = geometry(1.0 / g.qi).outer()
    return 1.0 - point.position[0], point.potential.value


DIRECTIONS = {
    "point": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "side": (0.0, 1.0, 0.0),
    "pole": (0.0, 0.0, 1.0),
}
AXES: dict[str, RadiusName] = {"+y": "side", "+z": "pole", "-x": "back"}


def direction(field: dict[str, Any], row: dict[str, str]) -> tuple[float, float, float]:
    """From a named radius, or a row's polar angle theta and azimuth phi in degrees."""
    if "radius" in field:
        return DIRECTIONS[field["radius"]]
    theta = math.radians(float(row[field["theta_column"]]))
    phi = math.radians(float(row[field["phi_column"]]))
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def evaluate(
    field: dict[str, Any],
    g: rocheometry.ComponentGeometry,
    m: float,
    conventions: dict[str, Any],
    row: dict[str, str],
) -> tuple[str, float]:
    """Return the catalogue key a column checks, and Rocheometry's value of it."""
    quantity = field["rocheometry_quantity"]
    frame = (
        field["frame"]
        if isinstance(field.get("frame"), dict)
        else conventions.get("frame")
    )
    spec = potential_spec(field, conventions)
    potential_key = (
        "potential.conventions.kopal" if spec["kind"].startswith("kopal") else None
    )
    if quantity == "l1_position_x":
        return "critical.l1.position", frame_x(g.l1().position[0], m, frame)
    if quantity == "l1_potential":
        value = in_convention(g.l1().potential.value, g.qi, spec)
        return potential_key or "critical.l1.potential", value
    if quantity == "l1_cone_angle_degrees":  # tan^2 phi = -U_xx / U_yy
        h = g.l1().hessian
        return "critical.l1.hessian", math.degrees(
            math.atan(math.sqrt(-h[0][0] / h[1][1]))
        )
    if quantity == "outer_position_x":
        return "critical.outer.position", frame_x(g.outer().position[0], m, frame)
    if quantity == "outer_distance":
        return "critical.outer.position", -g.outer().position[0]
    if quantity == "outer_potential":
        value = in_convention(g.outer().potential.value, g.qi, spec)
        return potential_key or "critical.outer.potential", value
    if quantity == "lagrange_x":
        return "critical.outer.position", frame_x(
            lagrange(field["point"], g)[0], m, frame
        )
    if quantity == "lagrange_potential":
        value = in_convention(lagrange(field["point"], g)[1], g.qi, spec)
        return potential_key or "critical.outer.potential", value
    if quantity == "contact_potential":  # Rocheometry's fill-out factor is F - 1
        s = g.surface(fill_out_factor=float(field["level"]["fill_out"]) - 1.0)
        return "selection.by_fill_out_factor", in_convention(
            s.potential.value, g.qi, spec
        )
    if quantity == "level_potential":
        value = in_convention(level(field, g, row), g.qi, spec)
        return potential_key or "critical.l1.potential", value
    if quantity == "named_radius":
        return "radii.named", g.lobe().named_radius(field["radius"])
    if quantity == "named_point_x":  # the lobe's axis point, as a position
        sign = -1.0 if field["radius"] == "back" else 1.0
        return "radii.named", frame_x(
            sign * g.lobe().named_radius(field["radius"]), m, frame
        )
    if quantity == "axis_crossing":  # the lobe's surface along an axis
        r = g.lobe().named_radius(AXES[field["axis"]])
        return "radii.named", frame_x(-r, m, frame) if field["axis"] == "-x" else r
    if quantity == "lobe_vertex":  # the farthest point of the section from the x-axis
        return "outlines.section.extent", g.lobe().section().bounds[3]
    if quantity == "level_radius":
        key = "radii.named" if "radius" in field else "radii.direction"
        return key, surface(g.qi, level(field, g, row)).radius(direction(field, row))
    if quantity == "lobe_volume":
        return "integrals.volume", g.lobe().volume()
    if quantity == "lobe_volume_radius":
        return "lobe.volume_radius", g.lobe_volume_radius()
    s = surface(g.qi, level(field, g, row))
    if quantity == "volume":
        return "integrals.volume", s.volume()
    if quantity == "volume_radius":
        return "integrals.volume", s.volume_radius()
    if quantity == "surface_area":
        return "integrals.area", s.area()
    if quantity == "mean_gravity":
        return "integrals.mean_gravity", scale(conventions) * s.mean_gravity()
    raise NotImplementedError(quantity)


# --- contact envelopes -----------------------------------------------------------
#
# Mochnacki divides a contact envelope by the plane through L1, Rocheometry along
# the gravity watershed, so only totals over both components compare. A
# primary's column names the partner tables it is combined with, read at the
# same row and column; the printed total's half-unit is the propagated
# rounding of every value in it.

ENVELOPE = (
    "envelope_volume",
    "envelope_max_volume",
    "envelope_area",
    "envelope_mean_gravity",
)


def partner_value(table: str, row: str, column: str) -> tuple[float, float]:
    """Return a value from another table, with its half-unit.

    The value is at a row, keyed by mass ratio, and a column of that table.
    """
    data = load(table)
    key = data["meta"]["conventions"]["mass_parameter"]["column"]
    text = next(r for r in data["rows"] if r[key] == row)[column]
    return float(text), half_unit(text)


def envelope(
    field: dict[str, Any],
    g: rocheometry.ComponentGeometry,
    conventions: dict[str, Any],
    row: dict[str, str],
) -> tuple[str, float, float, float]:
    """(key, Rocheometry's total, printed total, its half-unit) over both components."""
    quantity, column = field["rocheometry_quantity"], field["name"]
    components = (g, geometry(1.0 / g.qi))
    own = (float(row[column]), half_unit(row[column]))
    key = conventions["mass_parameter"]["column"]
    values = [own] + [partner_value(t, row[key], column) for t in field["partners"]]
    if quantity == "envelope_max_volume":  # at L2, which no surface selects
        radii = [
            c.max_volume_radius_fill() * c.lobe().volume_radius() for c in components
        ]
        return "lobe.max_fill.envelope_volume", *volume_from_radii(radii, values)
    # Rocheometry's fill-out factor is Mochnacki's F - 1.
    factor = float(field["level"]["fill_out"]) - 1.0
    parts = [c.surface(fill_out_factor=factor) for c in components]
    if quantity == "envelope_volume":
        radii = [p.volume_radius() for p in parts]
        return "integrals.volume.envelope", *volume_from_radii(radii, values)
    if quantity == "envelope_area":
        return (
            "integrals.area.envelope",
            sum(p.area() for p in parts),
            sum(v for v, _ in values),
            sum(h for _, h in values),
        )
    # Area-weighted mean gravity: own and partner gravities, then the two areas.
    (g1, h1), (g2, h2), (a1, k1), (a2, k2) = values
    area = a1 + a2
    expected = (g1 * a1 + g2 * a2) / area
    half = (
        a1 * h1 + a2 * h2 + abs(g1 - expected) * k1 + abs(g2 - expected) * k2
    ) / area
    actual = sum(p.mean_gravity() * p.area() for p in parts) / sum(
        p.area() for p in parts
    )
    return (
        "integrals.mean_gravity.envelope",
        scale(conventions) * actual,
        expected,
        half,
    )


def volume_from_radii(
    radii: list[float], values: list[tuple[float, float]]
) -> tuple[float, float, float]:
    """(Rocheometry's total volume, printed total, its half-unit) from volume radii."""
    expected = sum(4.0 / 3.0 * math.pi * r**3 for r, _ in values)
    half = sum(4.0 * math.pi * r**2 * h for r, h in values)
    return sum(4.0 / 3.0 * math.pi * r**3 for r in radii), expected, half


# --- comparison ------------------------------------------------------------------


def half_unit(text: str, significant_digits: int | None = None) -> float:
    """Return half a unit in the last printed digit.

    For tables that drop trailing zeros, the unit is in the last of
    `significant_digits` instead (where "1.8938" stands for 1.893800).
    """
    if significant_digits is None:
        exponent = Decimal(text).as_tuple().exponent
        assert isinstance(exponent, int), f"{text!r} is not a finite number"
    else:
        exponent = Decimal(text).copy_abs().adjusted() - significant_digits + 1
    return 0.5 * 10.0**exponent


def verdict(d: float, allowance: float, note: dict[str, Any] | None) -> str:
    if abs(d) <= 1.0:
        return "agrees"
    if abs(d) <= allowance:
        return "within allowance"
    if note and abs(d) <= note["max_half_units"]:
        return "explained"
    return "unexplained"
