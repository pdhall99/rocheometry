# Comparisons with published results

Do our definitions and conventions match the classical Roche model as others state it?
Each comparison uses expected values from someone else's statement of the model: their tables, asymptotic coefficients or identities.
That is what lets it catch a mismatch of definitions (potential normalization, mass ratio convention, which radius) that no check derived from our own definitions can.
These comparisons are verification, not validation: nothing here tests the model against observed stars, which is out of scope.

Each module holds one study, a `Study` class like the contract studies, and its rows are the same records, with the publication in the case's `source`.
The release run includes them; run one with `hatch run accuracy:study literature_tables`.
A comparison given only for context has rows with no bound and a note, so it never fails.

| Module | Compares with | Status |
| --- | --- | --- |
| `asymptotic_limits` | Guillochon & Prochaska's (2026) expansions of the lobe volume radius and its slope as qi → 0 and qi → ∞, and the fit's endpoint limits | done |
| `seidov` | Seidov's (2004) exact inverse formulas for the positions and potentials of L1 and the outer point, qi = 1e-8 to 1e8 | done |
| `eggleton` | Eggleton's (1983) lobe volume radius formula, to its stated 1%, from qi = 1e-8 to 1e8 | done |
| `literature_tables` | Published tables: Kuiper & Johnson, Kopal, Plavec & Kratochvil, Mochnacki, Leahy & Leahy, and Guillochon & Prochaska's reference solution (see [tables](tables/README.md)) | done |
