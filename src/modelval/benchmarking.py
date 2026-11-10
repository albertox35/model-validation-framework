"""Replica independiente, modelo retador y analisis de resultados.

SR 11-7 y PRA SS1/23 piden tres cosas distintas que suelen confundirse en una
sola, y cada una responde a una pregunta diferente:

**Replica independiente** (*independent replication*). ¿El codigo implementa lo
que la documentacion dice? Se reimplementa el modelo desde la especificacion, sin
mirar el codigo, y se comparan las salidas. Detecta errores de implementacion, no
de concepto.

**Modelo retador** (*challenger*). ¿La eleccion de modelo esta justificada? Se
construye una alternativa razonable --a menudo deliberadamente mas simple-- y se
compara su desempenio. Si el retador simple empata con el modelo en produccion,
la complejidad del segundo no esta justificada. Es la pregunta mas incomoda de
una validacion y la que mas valor aporta.

**Analisis de resultados** (*outcome analysis*). ¿Las predicciones se cumplieron?
Es backtesting, y es la unica evidencia empirica directa.

Un informe que solo trae replica esta diciendo «el codigo hace lo que dice el
documento», que no es lo mismo que «el modelo es adecuado».
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ReplicationResult",
    "ChallengerComparison",
    "OutcomeAnalysis",
    "replicate",
    "compare_challenger",
    "analyse_outcomes",
]


@dataclass(frozen=True)
class ReplicationResult:
    """Resultado de una replica independiente."""

    max_absolute_difference: float
    max_relative_difference: float
    rmse: float
    n_points: int
    tolerance: float
    n_exceeding: int

    @property
    def passed(self) -> bool:
        """``True`` si ninguna diferencia relativa excede la tolerancia."""
        return self.n_exceeding == 0

    def __str__(self) -> str:
        estado = "CONFORME" if self.passed else "DISCREPANCIA"
        return (
            f"replica {estado}: dif. relativa maxima {self.max_relative_difference:.3e} "
            f"(tolerancia {self.tolerance:.1e}), {self.n_exceeding}/{self.n_points} puntos exceden"
        )


def replicate(
    production_outputs: np.ndarray,
    replicated_outputs: np.ndarray,
    tolerance: float = 1e-6,
) -> ReplicationResult:
    """Compara la salida del modelo en produccion con una replica independiente.

    Se mide la diferencia **relativa** y no solo la absoluta: una tolerancia
    absoluta es inutilizable cuando la salida recorre varios ordenes de
    magnitud, que es el caso habitual de un precio o una exposicion.

    Parameters
    ----------
    production_outputs, replicated_outputs:
        Salidas a comparar, del mismo tamanio.
    tolerance:
        Tolerancia relativa.

    Returns
    -------
    ReplicationResult
    """
    a = np.asarray(production_outputs, dtype=float).ravel()
    b = np.asarray(replicated_outputs, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"las salidas deben tener la misma forma: {a.shape} y {b.shape}")
    if a.size == 0:
        raise ValueError("no hay puntos que comparar")

    absoluta = np.abs(a - b)
    escala = np.maximum(np.abs(a), 1e-12)
    relativa = absoluta / escala
    return ReplicationResult(
        max_absolute_difference=float(np.max(absoluta)),
        max_relative_difference=float(np.max(relativa)),
        rmse=float(np.sqrt(np.mean(absoluta**2))),
        n_points=int(a.size),
        tolerance=tolerance,
        n_exceeding=int(np.sum(relativa > tolerance)),
    )


@dataclass(frozen=True)
class ChallengerComparison:
    """Comparacion entre el modelo en produccion y un retador."""

    champion_error: float
    challenger_error: float
    champion_name: str = "produccion"
    challenger_name: str = "retador"
    metric_name: str = "RMSE"

    @property
    def improvement(self) -> float:
        """Mejora relativa del modelo en produccion sobre el retador.

        Positiva si el modelo en produccion es mejor.
        """
        if self.challenger_error == 0.0:
            return float("-inf") if self.champion_error > 0.0 else 0.0
        return float(1.0 - self.champion_error / self.challenger_error)

    def complexity_justified(self, threshold: float = 0.05) -> bool:
        """``True`` si el modelo en produccion supera al retador de forma material.

        Si devuelve ``False``, la complejidad adicional del modelo en produccion
        **no esta justificada por su desempenio**, que es una conclusion de
        validacion perfectamente legitima y rara vez bienvenida.
        """
        return self.improvement >= threshold

    def __str__(self) -> str:
        veredicto = (
            "complejidad justificada"
            if self.complexity_justified()
            else "COMPLEJIDAD NO JUSTIFICADA"
        )
        return (
            f"{self.metric_name}: {self.champion_name} {self.champion_error:.6f} frente a "
            f"{self.challenger_name} {self.challenger_error:.6f} "
            f"(mejora {self.improvement:+.1%}) -> {veredicto}"
        )


def compare_challenger(
    actual: np.ndarray,
    champion_predictions: np.ndarray,
    challenger_predictions: np.ndarray,
    champion_name: str = "produccion",
    challenger_name: str = "retador",
) -> ChallengerComparison:
    """Compara el desempenio predictivo de dos modelos sobre los mismos datos.

    Returns
    -------
    ChallengerComparison
    """
    y = np.asarray(actual, dtype=float).ravel()
    champ = np.asarray(champion_predictions, dtype=float).ravel()
    chall = np.asarray(challenger_predictions, dtype=float).ravel()
    if not (y.shape == champ.shape == chall.shape):
        raise ValueError("los tres vectores deben tener la misma forma")
    return ChallengerComparison(
        champion_error=float(np.sqrt(np.mean((y - champ) ** 2))),
        challenger_error=float(np.sqrt(np.mean((y - chall) ** 2))),
        champion_name=champion_name,
        challenger_name=challenger_name,
    )


@dataclass(frozen=True)
class OutcomeAnalysis:
    """Analisis de resultados: contraste de predicciones contra lo observado."""

    bias: float
    """Error medio con signo. Distinto de cero indica sesgo sistematico."""

    bias_t_statistic: float
    """Estadistico t del sesgo. Es lo que distingue un sesgo real del ruido."""

    rmse: float
    mean_absolute_error: float
    correlation: float
    n_observations: int

    @property
    def bias_is_significant(self) -> bool:
        """``True`` si el sesgo es estadisticamente distinto de cero al 5 %."""
        return abs(self.bias_t_statistic) > 1.96

    def __str__(self) -> str:
        marca = "SIGNIFICATIVO" if self.bias_is_significant else "no significativo"
        return (
            f"sesgo {self.bias:+.6f} (t = {self.bias_t_statistic:+.2f}, {marca}), "
            f"RMSE {self.rmse:.6f}, correlacion {self.correlation:.4f}, "
            f"n = {self.n_observations}"
        )


def analyse_outcomes(actual: np.ndarray, predicted: np.ndarray) -> OutcomeAnalysis:
    """Analiza el desempenio predictivo contra lo observado.

    El **estadistico t del sesgo** es la cifra que importa, no el sesgo a secas:
    un sesgo pequenio con muchas observaciones puede ser muy significativo, y uno
    grande con pocas puede ser ruido. Reportar solo la media del error invita a
    la conclusion equivocada en los dos sentidos.

    Returns
    -------
    OutcomeAnalysis
    """
    y = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError("actual y predicted deben tener la misma forma")
    if y.size < 3:
        raise ValueError("se requieren al menos 3 observaciones")

    error = p - y
    sesgo = float(np.mean(error))
    desv = float(np.std(error, ddof=1))
    t = sesgo / (desv / np.sqrt(error.size)) if desv > 0.0 else 0.0
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0.0 and np.std(p) > 0.0 else float("nan")

    return OutcomeAnalysis(
        bias=sesgo,
        bias_t_statistic=float(t),
        rmse=float(np.sqrt(np.mean(error**2))),
        mean_absolute_error=float(np.mean(np.abs(error))),
        correlation=corr,
        n_observations=int(y.size),
    )
