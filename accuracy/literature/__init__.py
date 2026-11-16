"""Comparisons with published results for the classical Roche model.

These are verification, not validation: they compare our solution of the
model with independent solutions of the same model by others (asymptotic
results, identities and tables). What they can catch that the contract
studies cannot is a mismatch of definitions or conventions, because the
expected values come from someone else's statement of the model.

Each module holds one study, a ``Study`` like those in ``accuracy/contract``
with ``kind = "literature"``, whose rows name the publication in each case's
``source``. A comparison given for context only has rows with no bound and a
note. They run in the release run, after the contract studies.
"""
