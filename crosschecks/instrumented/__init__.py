"""Instrumented checks of the library's own arithmetic.

Unlike the rest of `crosschecks`, these modules are not independent truths. They
patch or read Rocheometry internals to replay its node arithmetic with error
bounds, and to measure its quadratures afresh.
"""
