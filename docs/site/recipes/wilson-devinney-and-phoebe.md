---
description: "Convert a Wilson–Devinney or PHOEBE solution to Rocheometry: the reciprocal mass ratio and a Kopal potential select both components."
---

# Convert a Wilson–Devinney or PHOEBE solution

Select both components from a fitted mass ratio and potential.
For a contact binary with Wilson–Devinney or PHOEBE values `q = 0.3` and `Ω = 2.4`, pass the reciprocal mass ratio and a Kopal potential for component 1:

```pycon
>>> import rocheometry
>>>
>>> binary = rocheometry.BinaryGeometry(q=1 / 0.3)
>>> level = rocheometry.Potential.kopal(2.4, component_id=1)
>>> surface1 = binary.component(1).surface(potential=level)  # WD and PHOEBE primary
>>> surface2 = binary.component(2).surface(potential=level)  # WD and PHOEBE secondary
>>> surface1.regime, round(surface1.fill_out_factor(), 3)
(<LevelRegime.CONTACT: 'contact'>, 0.354)
>>> surface2.fill_out_factor() == surface1.fill_out_factor()
True
>>> kopal = rocheometry.PotentialConvention.KOPAL
>>> round(
...     binary.convert_potential(
...         binary.l1().potential, convention=kopal, component_id=1
...     ).value,
...     4,
... )
2.4662

```

Both surfaces have the same fill-out factor, about 0.354.
The final value, 2.4662, is the critical Kopal potential at L1, for comparison with the fit.
Wilson–Devinney and PHOEBE use `M2/M1`; [potential conventions](../guide/potentials.md#from-wilsondevinney-or-phoebe) explains the conversion.
