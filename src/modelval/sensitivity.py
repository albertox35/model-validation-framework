"""Analisis de sensibilidad global: indices de Sobol y efectos elementales de Morris.

«Analisis de sensibilidad» a secas no significa nada. Hay dos familias y sirven
para cosas distintas:

**Sensibilidad local** --derivadas parciales en un punto-- responde «como cambia
la salida si muevo un poco este parametro desde aqui». Es lo que da una griega.
No sirve para validar un modelo no lineal, porque el resultado depende del punto
elegido y no dice nada sobre interacciones.

**Sensibilidad global** explora todo el espacio de parametros a la vez. Dos
metodos, en el orden en que se usan:

1. **Morris** (efectos elementales) para **cribado**. Barato, ``O(d)``
   evaluaciones por trayectoria, identifica que parametros son irrelevantes para
   descartarlos. Distingue efectos no lineales o con interaccion mediante la
   desviacion estandar de los efectos.
2. **Sobol** (descomposicion de varianza) para **cuantificar**. Caro,
   ``O(N(d+2))`` evaluaciones, pero responde exactamente «que fraccion de la
   varianza de la salida explica cada parametro», y la diferencia entre el indice
   total y el de primer orden mide las interacciones.

La implementacion de Sobol se valida contra la **funcion de Ishigami**, cuyos
indices tienen forma cerrada. Es el banco de pruebas estandar del area
precisamente porque tiene un parametro con indice de primer orden **cero** y un
indice total **positivo**: solo actua a traves de una interaccion. Un estimador
mal implementado suele acertar los dos primeros indices y fallar justo ahi.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

__all__ = [
    "SobolIndices",
    "MorrisResult",
    "sobol_indices",
    "morris_screening",
    "ishigami",
    "ishigami_analytic_indices",
]


@dataclass(frozen=True)
class SobolIndices:
    """Indices de sensibilidad de Sobol."""

    first_order: np.ndarray
    """``S_i``: fraccion de varianza explicada por el parametro ``i`` solo."""

    total_order: np.ndarray
    """``S_Ti``: fraccion explicada por ``i`` y todas sus interacciones."""

    first_order_stderr: np.ndarray
    total_order_stderr: np.ndarray
    variance: float
    n_samples: int
    n_evaluations: int
    parameter_names: tuple[str, ...] | None = None

    @property
    def interaction_strength(self) -> np.ndarray:
        """``S_Ti - S_i``: la parte de la influencia que pasa por interacciones.

        Un parametro con ``S_i`` cercano a cero y ``S_Ti`` grande **solo** actua
        en combinacion con otros. Es exactamente lo que un analisis local, de una
        variable a la vez, no puede detectar.
        """
        return self.total_order - self.first_order

    @property
    def sum_first_order(self) -> float:
        """Suma de los indices de primer orden.

        Vale 1 si el modelo es aditivo. Cuanto mas por debajo de 1, mas peso
        tienen las interacciones.
        """
        return float(np.sum(self.first_order))

    def ranking(self) -> list[tuple[str, float, float]]:
        """Parametros ordenados por indice total, de mayor a menor."""
        nombres = self.parameter_names or tuple(
            f"x{i + 1}" for i in range(self.first_order.size)
        )
        filas = list(zip(nombres, self.first_order, self.total_order))
        return sorted(filas, key=lambda f: f[2], reverse=True)


def sobol_indices(
    model: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    n_samples: int = 8192,
    seed: int = 20260101,
    parameter_names: tuple[str, ...] | None = None,
) -> SobolIndices:
    """Indices de Sobol por los estimadores de Saltelli y Jansen.

    Se usan dos estimadores distintos a proposito, porque cada uno es el mejor
    para su indice (Saltelli et al., 2010):

    - **Primer orden**, estimador de Saltelli:

      .. math::

          S_i = \\frac{1}{N V}\\sum_{j}
                f(B_j)\\left[f(A_B^{(i)})_j - f(A_j)\\right]

    - **Orden total**, estimador de Jansen:

      .. math::

          S_{Ti} = \\frac{1}{2NV}\\sum_{j}
                   \\left[f(A_j) - f(A_B^{(i)})_j\\right]^2

    El de Jansen para el indice total es claramente superior al estimador clasico
    cuando el indice es pequenio: el clasico resta dos cantidades casi iguales y
    el error relativo se dispara, mientras el de Jansen es una suma de cuadrados
    y no sufre cancelacion.

    Parameters
    ----------
    model:
        Funcion vectorizada que toma una matriz ``(n, d)`` y devuelve ``(n,)``.
    bounds:
        Matriz ``(d, 2)`` con el minimo y el maximo de cada parametro. El
        muestreo es uniforme en la caja.
    n_samples:
        Tamanio ``N`` de cada matriz base. Potencia de dos, porque el muestreo
        usa una red de Sobol.
    seed:
        Semilla.

    Returns
    -------
    SobolIndices
    """
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds debe ser una matriz (d, 2)")
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("cada cota superior debe exceder a la inferior")
    d = b.shape[0]
    if n_samples & (n_samples - 1):
        raise ValueError(
            f"n_samples = {n_samples} no es potencia de dos; el muestreo usa una red "
            "de Sobol, cuyas propiedades de equidistribucion se dan en bloques de 2^m"
        )
    if parameter_names is not None and len(parameter_names) != d:
        raise ValueError("parameter_names debe tener un nombre por parametro")

    # Se muestrean 2d dimensiones de una sola red para que A y B sean
    # independientes entre si pero cada uno bien equidistribuido.
    motor = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    puntos = motor.random(n_samples)
    escala = b[:, 1] - b[:, 0]
    a = b[:, 0] + puntos[:, :d] * escala
    b_mat = b[:, 0] + puntos[:, d:] * escala

    f_a = np.asarray(model(a), dtype=float).ravel()
    f_b = np.asarray(model(b_mat), dtype=float).ravel()
    if f_a.size != n_samples:
        raise ValueError(
            f"el modelo devolvio {f_a.size} valores para {n_samples} filas; "
            "debe estar vectorizado y devolver un valor por fila"
        )

    varianza = float(np.var(np.concatenate([f_a, f_b]), ddof=1))
    if varianza <= 0.0:
        raise ValueError("la salida del modelo no tiene varianza: no hay nada que descomponer")

    s1 = np.empty(d)
    st = np.empty(d)
    s1_se = np.empty(d)
    st_se = np.empty(d)

    for i in range(d):
        a_bi = a.copy()
        a_bi[:, i] = b_mat[:, i]
        f_abi = np.asarray(model(a_bi), dtype=float).ravel()

        # Saltelli para el primer orden.
        muestra_s1 = f_b * (f_abi - f_a) / varianza
        s1[i] = float(np.mean(muestra_s1))
        s1_se[i] = float(np.std(muestra_s1, ddof=1) / np.sqrt(n_samples))

        # Jansen para el orden total: suma de cuadrados, sin cancelacion.
        muestra_st = 0.5 * (f_a - f_abi) ** 2 / varianza
        st[i] = float(np.mean(muestra_st))
        st_se[i] = float(np.std(muestra_st, ddof=1) / np.sqrt(n_samples))

    return SobolIndices(
        first_order=s1,
        total_order=st,
        first_order_stderr=s1_se,
        total_order_stderr=st_se,
        variance=varianza,
        n_samples=n_samples,
        n_evaluations=n_samples * (d + 2),
        parameter_names=parameter_names,
    )


@dataclass(frozen=True)
class MorrisResult:
    """Efectos elementales de Morris."""

    mu_star: np.ndarray
    """Media del valor absoluto de los efectos elementales: influencia global."""

    sigma: np.ndarray
    """Desviacion estandar de los efectos: no linealidad o interaccion."""

    mu: np.ndarray
    """Media con signo. Si difiere mucho de ``mu_star``, el efecto cambia de
    signo segun la region, lo que es una senial de no monotonia."""

    n_trajectories: int
    n_evaluations: int
    parameter_names: tuple[str, ...] | None = None

    def screen(self, threshold_fraction: float = 0.05) -> list[str]:
        """Parametros cuya influencia queda por debajo del umbral.

        Son candidatos a fijarse como constantes, lo que reduce la dimension del
        analisis de Sobol posterior, que es el caro.

        Parameters
        ----------
        threshold_fraction:
            Fraccion del ``mu_star`` maximo por debajo de la cual se considera
            despreciable.
        """
        nombres = self.parameter_names or tuple(f"x{i + 1}" for i in range(self.mu_star.size))
        umbral = threshold_fraction * float(np.max(self.mu_star))
        return [n for n, m in zip(nombres, self.mu_star) if m < umbral]

    def ranking(self) -> list[tuple[str, float, float]]:
        """Parametros ordenados por ``mu_star``."""
        nombres = self.parameter_names or tuple(f"x{i + 1}" for i in range(self.mu_star.size))
        return sorted(zip(nombres, self.mu_star, self.sigma), key=lambda f: f[1], reverse=True)


def morris_screening(
    model: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    n_trajectories: int = 100,
    n_levels: int = 8,
    seed: int = 20260101,
    parameter_names: tuple[str, ...] | None = None,
) -> MorrisResult:
    """Cribado por efectos elementales de Morris.

    Cada trayectoria parte de un punto aleatorio de una malla y mueve **un
    parametro a la vez**, registrando el cambio en la salida. Con ``d``
    parametros, una trayectoria cuesta ``d + 1`` evaluaciones y entrega ``d``
    efectos elementales.

    La virtud del metodo es la relacion costo-informacion: con ``100(d+1)``
    evaluaciones se identifica que parametros son irrelevantes, frente a las
    ``N(d+2)`` con ``N`` en los miles que exige Sobol. El flujo correcto es
    cribar con Morris y cuantificar con Sobol solo lo que sobrevive.

    Returns
    -------
    MorrisResult
    """
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds debe ser una matriz (d, 2)")
    d = b.shape[0]
    if n_levels < 4 or n_levels % 2 != 0:
        raise ValueError("n_levels debe ser par y al menos 4")
    if n_trajectories < 2:
        raise ValueError("se requieren al menos 2 trayectorias")

    rng = np.random.default_rng(seed)
    delta = n_levels / (2.0 * (n_levels - 1.0))
    escala = b[:, 1] - b[:, 0]
    efectos = np.empty((n_trajectories, d))

    for k in range(n_trajectories):
        # Punto base en la malla, dejando sitio para el salto.
        base = rng.integers(0, n_levels // 2, size=d) / (n_levels - 1.0)
        orden = rng.permutation(d)
        signos = rng.choice([-1.0, 1.0], size=d)

        puntos = np.empty((d + 1, d))
        actual = base.copy()
        puntos[0] = actual
        for paso, i in enumerate(orden):
            actual = actual.copy()
            # Si el salto saliera del cubo unidad se **invierte el signo**, no se
            # recorta. Recortar produce saltos de longitud distinta a delta y, en
            # el borde, saltos de longitud cero: el efecto elemental deja de
            # estar definido y hay que imputarlo como nulo. Eso contamina la
            # desviacion estandar de los efectos con ceros espurios, y es justo
            # esa desviacion la que mide la no linealidad. Un parametro con
            # efecto perfectamente lineal pasaria a tener sigma > 0.
            candidato = actual[i] + signos[paso] * delta
            if not 0.0 <= candidato <= 1.0:
                candidato = actual[i] - signos[paso] * delta
            actual[i] = candidato
            puntos[paso + 1] = actual

        reales = b[:, 0] + puntos * escala
        salidas = np.asarray(model(reales), dtype=float).ravel()
        for paso, i in enumerate(orden):
            salto = puntos[paso + 1, i] - puntos[paso, i]
            if salto == 0.0:
                efectos[k, i] = 0.0
            else:
                efectos[k, i] = (salidas[paso + 1] - salidas[paso]) / salto

    return MorrisResult(
        mu_star=np.mean(np.abs(efectos), axis=0),
        sigma=np.std(efectos, axis=0, ddof=1),
        mu=np.mean(efectos, axis=0),
        n_trajectories=n_trajectories,
        n_evaluations=n_trajectories * (d + 1),
        parameter_names=parameter_names,
    )


# --------------------------------------------------- banco de pruebas estandar


def ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """Funcion de Ishigami, el banco de pruebas estandar del area.

    .. math:: f(x) = \\sin x_1 + a\\sin^2 x_2 + b\\,x_3^4\\sin x_1

    con :math:`x_i \\sim U(-\\pi, \\pi)`.

    Su valor como prueba esta en la estructura de ``x3``: **no aparece solo** en
    ninguna parte, solo multiplicando a ``sin x1``. Por tanto su indice de primer
    orden es **exactamente cero** y su indice total es **positivo**. Un estimador
    mal implementado acierta los indices de ``x1`` y ``x2`` --que son grandes-- y
    falla justo en ese contraste, que es el que distingue un analisis global de
    uno local.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 3:
        raise ValueError("la funcion de Ishigami toma exactamente 3 parametros")
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def ishigami_analytic_indices(a: float = 7.0, b: float = 0.1) -> dict[str, np.ndarray]:
    """Indices de Sobol analiticos de la funcion de Ishigami.

    Con :math:`a = 7` y :math:`b = 0.1` los valores son
    ``S = (0.3139, 0.4424, 0.0000)`` y
    ``S_T = (0.5576, 0.4424, 0.2437)``.

    Returns
    -------
    dict
        Con las claves ``variance``, ``first_order`` y ``total_order``.
    """
    pi4 = np.pi**4
    pi8 = np.pi**8
    v1 = 0.5 * (1.0 + b * pi4 / 5.0) ** 2
    v2 = a**2 / 8.0
    v3 = 0.0
    v13 = b**2 * pi8 * (1.0 / 18.0 - 1.0 / 50.0)
    varianza = v2 + b * pi4 / 5.0 + b**2 * pi8 / 18.0 + 0.5
    return {
        "variance": np.array(varianza),
        "first_order": np.array([v1, v2, v3]) / varianza,
        "total_order": np.array([v1 + v13, v2, v13]) / varianza,
    }
