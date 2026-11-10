"""Framework de validacion independiente de modelos cuantitativos.

Cuatro bloques, mapeados a las secciones de SR 11-7 y PRA SS1/23:

- :mod:`modelval.sensitivity` analisis de sensibilidad **global**: indices de
  Sobol por los estimadores de Saltelli y Jansen, y cribado por efectos
  elementales de Morris. Validado contra los indices analiticos de la funcion
  de Ishigami.
- :mod:`modelval.stability` estabilidad poblacional (PSI), prueba de
  Kolmogorov-Smirnov y deriva de la **relacion** entre entrada y salida.
- :mod:`modelval.benchmarking` replica independiente, modelo retador y analisis
  de resultados, que son tres cosas distintas y suelen confundirse.
- :mod:`modelval.report` informe estructurado **por seccion del marco
  regulatorio**, no por tipo de prueba.
"""

from __future__ import annotations

from .benchmarking import (
    ChallengerComparison,
    OutcomeAnalysis,
    ReplicationResult,
    analyse_outcomes,
    compare_challenger,
    replicate,
)
from .report import FRAMEWORK_SECTIONS, Finding, Severity, ValidationReport
from .sensitivity import (
    MorrisResult,
    SobolIndices,
    ishigami,
    ishigami_analytic_indices,
    morris_screening,
    sobol_indices,
)
from .stability import (
    StabilityReport,
    characteristic_stability,
    ks_drift_test,
    population_stability_index,
    relationship_drift,
)

__version__ = "0.1.0"

__all__ = [
    "FRAMEWORK_SECTIONS",
    "ChallengerComparison",
    "Finding",
    "MorrisResult",
    "OutcomeAnalysis",
    "ReplicationResult",
    "Severity",
    "SobolIndices",
    "StabilityReport",
    "ValidationReport",
    "__version__",
    "analyse_outcomes",
    "characteristic_stability",
    "compare_challenger",
    "ishigami",
    "ishigami_analytic_indices",
    "ks_drift_test",
    "morris_screening",
    "population_stability_index",
    "relationship_drift",
    "replicate",
    "sobol_indices",
]
