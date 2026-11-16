"""Mass ratios and fractions on canonical Roche-domain inputs."""

from rocheometry._types import ComponentId


def component_mass_ratio(q: float, component_id: ComponentId) -> float:
    """Return q_i = M_i/M_companion from the system mass ratio M1/M2.

    Args:
        q: System mass ratio `M1/M2`; must be finite and positive.
        component_id: Component number, either `1` or `2`.
    """
    return q if component_id == 1 else 1.0 / q


def mass_fractions(qi: float) -> tuple[float, float]:
    """Return selected-component and companion fractions of the total mass.

    qi is the finite positive component-local ratio M_component/M_companion.

    Args:
        qi: Component-to-companion mass ratio; must be finite and positive.

    Returns:
        The `(component, companion)` fractions of total mass.
    """
    companion = 1.0 / (1.0 + qi)
    component = qi * companion
    return (component, companion)
