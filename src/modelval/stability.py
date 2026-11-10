"""Estabilidad poblacional y deteccion de deriva.

Un modelo se degrada sin que nada falle. Los datos de entrada se desplazan, la
relacion entre variables cambia, y el modelo sigue produciendo numeros con la
misma apariencia. Detectarlo es parte central de la validacion continua que
exigen SR 11-7 y PRA SS1/23.

Tres diagnosticos complementarios, que miden cosas distintas:

**PSI** (*population stability index*) compara la **distribucion** de una
variable entre dos periodos. Es el estandar de facto en banca de consumo. Sus
umbrales convencionales son 0.10 y 0.25, y conviene saber que son
**convencionales, no estadisticos**: no corresponden a ningun nivel de
significancia.

**Kolmogorov-Smirnov** compara las distribuciones con una prueba de hipotesis
propiamente dicha, con p-valor. A diferencia del PSI, su potencia crece con el
tamanio de muestra, lo que tiene una consecuencia practica incomoda: con
millones de observaciones rechaza desplazamientos irrelevantes.

**Deriva de la relacion** mira si cambio el vinculo entre entrada y salida, no
solo la distribucion de la entrada. Es la deriva que mas importa y la que el PSI
no puede ver: las variables pueden mantener su distribucion mientras el modelo
deja de predecir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "StabilityReport",
    "population_stability_index",
    "ks_drift_test",
    "relationship_drift",
    "characteristic_stability",
]

_EPS = 1e-10


@dataclass(frozen=True)
class StabilityReport:
    """Diagnostico de estabilidad de una variable."""

    name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    n_reference: int
    n_current: int

    @property
    def psi_verdict(self) -> str:
        """Lectura convencional del PSI.

        Los umbrales 0.10 y 0.25 son convencion de la industria, **no** niveles
        de significancia: no hay una distribucion nula detras. Se reportan como
        guia, no como prueba.
        """
        if self.psi < 0.10:
            return "estable"
        if self.psi < 0.25:
            return "desplazamiento moderado"
        return "desplazamiento significativo"

    def ks_rejects(self, significance: float = 0.05) -> bool:
        """``True`` si Kolmogorov-Smirnov rechaza la igualdad de distribuciones."""
        return self.ks_p_value < significance

    def __str__(self) -> str:
        return (
            f"{self.name}: PSI = {self.psi:.4f} ({self.psi_verdict}), "
            f"KS = {self.ks_statistic:.4f} (p = {self.ks_p_value:.4f})"
        )


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    bin_edges: np.ndarray | None = None,
) -> float:
    """Indice de estabilidad poblacional.

    .. math::

        PSI = \\sum_b \\left(p_b^{cur} - p_b^{ref}\\right)
              \\ln\\frac{p_b^{cur}}{p_b^{ref}}

    Es la divergencia de Jeffreys, es decir la simetrizacion de
    Kullback-Leibler, asi que ``PSI(a,b) = PSI(b,a)`` **con los mismos tramos**.

    La salvedad importa en la practica: si los tramos se derivan por percentiles
    de la muestra de referencia --que es lo habitual y el comportamiento por
    omision aqui-- intercambiar los papeles cambia los tramos y la simetria se
    vuelve aproximada. Para comparaciones encadenadas entre validaciones
    sucesivas conviene fijar ``bin_edges`` una vez y reutilizarlos, con lo que
    los valores si son comparables entre si y la simetria es exacta.

    Parameters
    ----------
    reference:
        Muestra de referencia, normalmente el periodo de desarrollo.
    current:
        Muestra actual.
    n_bins:
        Numero de tramos. Se construyen por **percentiles de la referencia**, no
        equiespaciados: con una variable asimetrica los tramos equiespaciados
        dejan tramos vacios y el PSI se vuelve inestable.
    bin_edges:
        Bordes explicitos, si se quieren fijar entre validaciones sucesivas.

    Returns
    -------
    float
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.size < 10 or cur.size < 10:
        raise ValueError("se requieren al menos 10 observaciones en cada muestra")

    if bin_edges is None:
        cuantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bordes = np.quantile(ref, cuantiles)
        bordes[0], bordes[-1] = -np.inf, np.inf
        bordes = np.unique(bordes)
        if bordes.size < 3:
            raise ValueError(
                "la referencia no tiene suficiente dispersion para construir tramos; "
                "probablemente sea casi constante"
            )
    else:
        bordes = np.asarray(bin_edges, dtype=float)

    conteo_ref, _ = np.histogram(ref, bins=bordes)
    conteo_cur, _ = np.histogram(cur, bins=bordes)
    p_ref = np.maximum(conteo_ref / ref.size, _EPS)
    p_cur = np.maximum(conteo_cur / cur.size, _EPS)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def ks_drift_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Prueba de Kolmogorov-Smirnov de dos muestras.

    Returns
    -------
    estadistico, p_valor
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    resultado = stats.ks_2samp(ref, cur)
    return float(resultado.statistic), float(resultado.pvalue)


def characteristic_stability(
    reference: np.ndarray,
    current: np.ndarray,
    names: tuple[str, ...] | None = None,
    n_bins: int = 10,
) -> list[StabilityReport]:
    """Estabilidad variable por variable de una matriz de caracteristicas.

    Parameters
    ----------
    reference, current:
        Matrices ``(n, d)`` con las mismas columnas.

    Returns
    -------
    list[StabilityReport]
        Ordenados por PSI descendente, para que lo mas desplazado aparezca
        primero.
    """
    ref = np.atleast_2d(np.asarray(reference, dtype=float))
    cur = np.atleast_2d(np.asarray(current, dtype=float))
    if ref.shape[1] != cur.shape[1]:
        raise ValueError(
            f"referencia con {ref.shape[1]} variables y actual con {cur.shape[1]}: "
            "deben coincidir"
        )
    d = ref.shape[1]
    etiquetas = names or tuple(f"v{i + 1}" for i in range(d))
    if len(etiquetas) != d:
        raise ValueError("names debe tener un nombre por variable")

    informes = []
    for i, nombre in enumerate(etiquetas):
        psi = population_stability_index(ref[:, i], cur[:, i], n_bins)
        ks, p = ks_drift_test(ref[:, i], cur[:, i])
        informes.append(
            StabilityReport(
                name=nombre,
                psi=psi,
                ks_statistic=ks,
                ks_p_value=p,
                n_reference=int(ref.shape[0]),
                n_current=int(cur.shape[0]),
            )
        )
    return sorted(informes, key=lambda r: r.psi, reverse=True)


def relationship_drift(
    reference_features: np.ndarray,
    reference_target: np.ndarray,
    current_features: np.ndarray,
    current_target: np.ndarray,
) -> dict[str, float]:
    """Deriva de la **relacion** entre entrada y salida.

    Es la deriva que mas importa y la que el PSI no puede detectar: las variables
    pueden conservar exactamente su distribucion mientras el vinculo con el
    objetivo se invierte. En ese caso el PSI vale cero y el modelo esta roto.

    Se ajusta una regresion lineal en cada periodo y se compara el cambio
    relativo de los coeficientes y la perdida de ajuste.

    Returns
    -------
    dict
        Con ``max_coefficient_change``, ``mean_coefficient_change``,
        ``r2_reference``, ``r2_current`` y ``r2_degradation``.
    """
    xr = np.atleast_2d(np.asarray(reference_features, dtype=float))
    yr = np.asarray(reference_target, dtype=float).ravel()
    xc = np.atleast_2d(np.asarray(current_features, dtype=float))
    yc = np.asarray(current_target, dtype=float).ravel()
    if xr.shape[1] != xc.shape[1]:
        raise ValueError("las matrices de caracteristicas deben tener las mismas columnas")
    if xr.shape[0] != yr.size or xc.shape[0] != yc.size:
        raise ValueError("el numero de filas no coincide con el del objetivo")

    def ajusta(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        disenio = np.column_stack([np.ones(x.shape[0]), x])
        coef, *_ = np.linalg.lstsq(disenio, y, rcond=None)
        residuo = y - disenio @ coef
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residuo**2)) / total if total > 0.0 else float("nan")
        return coef, r2

    coef_ref, r2_ref = ajusta(xr, yr)
    coef_cur, r2_cur = ajusta(xc, yc)

    # El denominador del cambio relativo se acota por abajo con una fraccion del
    # coeficiente mas grande del modelo. Sin ese suelo, un coeficiente cuyo
    # valor verdadero es cero --tipicamente la constante-- se estima como un
    # numero diminuto distinto en cada periodo, y el cambio relativo frente a el
    # resulta enorme. El diagnostico se dispararia por el coeficiente que menos
    # importa, que es el fallo clasico de medir cambios en terminos relativos.
    suelo = 0.01 * float(np.max(np.abs(coef_ref)))
    escala = np.maximum(np.abs(coef_ref), max(suelo, _EPS))
    cambio = np.abs(coef_cur - coef_ref) / escala

    return {
        "max_coefficient_change": float(np.max(cambio)),
        "mean_coefficient_change": float(np.mean(cambio)),
        "r2_reference": r2_ref,
        "r2_current": r2_cur,
        "r2_degradation": float(r2_ref - r2_cur),
    }
