"""Validacion completa de un modelo, con informe mapeado al marco regulatorio.

Ejecutar con:

    python examples/validate_model.py
"""

from __future__ import annotations

from datetime import date

import numpy as np

from modelval import (
    FRAMEWORK_SECTIONS,
    Finding,
    Severity,
    ValidationReport,
    analyse_outcomes,
    characteristic_stability,
    compare_challenger,
    ishigami,
    ishigami_analytic_indices,
    morris_screening,
    population_stability_index,
    relationship_drift,
    replicate,
    sobol_indices,
)

BOUNDS = np.array([[-np.pi, np.pi]] * 3)


def seccion(titulo: str) -> None:
    print(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}")


def demo_sobol() -> None:
    seccion("Sensibilidad global: Sobol contra los valores analiticos de Ishigami")
    an = ishigami_analytic_indices()
    print("La funcion de Ishigami es el banco de pruebas estandar porque x3 **no**")
    print("aparece sola: su indice de primer orden es exactamente cero y su indice")
    print("total es positivo. Solo actua por interaccion.\n")
    print(f"{'N':>7} {'S1':>8} {'S2':>8} {'S3':>9}   {'ST1':>8} {'ST2':>8} {'ST3':>8}")
    for n in (1_024, 4_096, 16_384, 65_536):
        r = sobol_indices(ishigami, BOUNDS, n_samples=n, seed=7)
        fila = " ".join(f"{v:>8.4f}" for v in r.first_order)
        fila2 = " ".join(f"{v:>8.4f}" for v in r.total_order)
        print(f"{n:>7} {fila}   {fila2}")
    print(f"{'exacto':>7} " + " ".join(f"{v:>8.4f}" for v in an["first_order"]) + "   "
          + " ".join(f"{v:>8.4f}" for v in an["total_order"]))

    r = sobol_indices(ishigami, BOUNDS, n_samples=65_536, seed=7, parameter_names=("x1", "x2", "x3"))
    print(f"\n  error maximo en indices totales: {np.max(np.abs(r.total_order - an['total_order'])):.1e}")
    print(f"  interaccion (ST - S): {np.round(r.interaction_strength, 4)}")
    print(f"  suma de indices de primer orden: {r.sum_first_order:.4f} (< 1 => hay interacciones)")
    print(f"  evaluaciones del modelo: {r.n_evaluations:,}")


def demo_morris() -> None:
    seccion("Cribado con Morris: identificar lo irrelevante antes de cuantificar")

    def modelo(x: np.ndarray) -> np.ndarray:
        return 5.0 * x[:, 0] + np.sin(10.0 * x[:, 1]) + 0.0 * x[:, 2]

    cotas = np.array([[0.0, 1.0]] * 3)
    m = morris_screening(
        modelo, cotas, n_trajectories=300, seed=5, parameter_names=("lineal", "no_lineal", "nulo")
    )
    print(f"{'parametro':>12} {'mu*':>10} {'sigma':>10}  lectura")
    for nombre, mu, sigma in m.ranking():
        if mu < 1e-10:
            lectura = "irrelevante: puede fijarse"
        elif sigma < 1e-8:
            lectura = "efecto lineal"
        else:
            lectura = "no lineal o con interaccion"
        print(f"{nombre:>12} {mu:>10.4f} {sigma:>10.4f}  {lectura}")
    print(f"\n  candidatos a fijar: {m.screen()}")
    s = sobol_indices(modelo, cotas, n_samples=8_192, seed=5)
    print(f"  evaluaciones: Morris {m.n_evaluations:,} frente a Sobol {s.n_evaluations:,} "
          f"({s.n_evaluations / m.n_evaluations:.0f}x)")
    print("\n  sigma = 0 significa efecto identico en todo el dominio. Es informacion")
    print("  que mu* sola no da, y la razon de reportar las dos.")


def demo_estabilidad() -> None:
    seccion("Vigilancia continua: la deriva que el PSI no puede ver")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5_000, 2))
    y_ref = 2.0 * x[:, 0] + x[:, 1] + 0.1 * rng.normal(size=5_000)
    y_cur = -2.0 * x[:, 0] + x[:, 1] + 0.1 * rng.normal(size=5_000)

    print("Se invierte el signo de la relacion entre x1 y el objetivo, manteniendo")
    print("EXACTAMENTE la misma distribucion de las entradas.\n")
    for i in range(2):
        print(f"  PSI de x{i + 1}: {population_stability_index(x[:, i], x[:, i]):.6f}")
    print("  -> el PSI no detecta nada, porque las marginales no cambiaron\n")

    d = relationship_drift(x, y_ref, x, y_cur)
    print(f"  cambio maximo de coeficientes: {d['max_coefficient_change']:.4f}")
    print(f"  R2 referencia / actual       : {d['r2_reference']:.4f} / {d['r2_current']:.4f}")
    print("  -> cada periodo se ajusta bien por separado; el problema es que cambio.")
    print("     Un modelo entrenado en el primero fallaria por completo en el segundo.")

    print("\nDesplazamiento de las entradas, que el PSI si detecta:")
    ref = rng.normal(size=(10_000, 3))
    cur = np.column_stack(
        [rng.normal(0.0, 1.0, 10_000), rng.normal(1.5, 1.0, 10_000), rng.normal(0.3, 1.0, 10_000)]
    )
    for inf in characteristic_stability(ref, cur, names=("estable", "muy_movida", "poco_movida")):
        print(f"  {inf}")


def demo_benchmarking() -> None:
    seccion("Replica, retador y analisis de resultados: tres cosas distintas")
    rng = np.random.default_rng(3)
    y = rng.normal(size=2_000)

    print("1) REPLICA: el codigo implementa lo que dice la documentacion?")
    a = np.array([1e-6, 1.0, 1e6])
    print(f"   {replicate(a, a * 1.001, tolerance=1e-6)}")
    print("   Se mide diferencia RELATIVA: una tolerancia absoluta es inutilizable")
    print("   cuando la salida recorre varios ordenes de magnitud.\n")

    print("2) RETADOR: la eleccion de modelo esta justificada?")
    ruido = rng.normal(size=2_000)
    print(f"   {compare_challenger(y, y + 0.50 * ruido, y + 0.51 * ruido)}")
    print(f"   {compare_challenger(y, y + 0.20 * ruido, y + 1.00 * ruido)}")
    print("   Si un retador simple empata, la complejidad no esta justificada.\n")

    print("3) ANALISIS DE RESULTADOS: las predicciones se cumplieron?")
    for n in (20, 5_000):
        muestra = rng.normal(size=n)
        res = analyse_outcomes(muestra, muestra + 0.1 + 0.5 * rng.normal(size=n))
        print(f"   n = {n:>5}: {res}")
    print("   El MISMO sesgo de 0.1 es ruido con 20 observaciones y significativo")
    print("   con 5 000. Por eso se reporta el estadistico t, no solo la media.")


def demo_informe() -> None:
    seccion("Informe: una seccion sin evaluar no es una seccion cumplida")
    r = ValidationReport(
        model_name="xva-engine / motor de CVA",
        model_owner="Tesoreria",
        validator="Validacion independiente de modelos",
        validation_date=date(2026, 11, 9),
        model_use="calculo de CVA para reporte regulatorio y ajuste de precios",
        risk_tier="alto",
    )
    print(f"Informe recien abierto: {r.conclusion}")
    print(f"  secciones del marco sin evidencia: {len(r.sections_without_evidence)} de "
          f"{len(FRAMEWORK_SECTIONS)}\n")

    r.add(
        Finding(
            "analisis_de_sensibilidad",
            Severity.MEDIUM,
            "interaccion no documentada en la especificacion",
            "El analisis de Sobol muestra que un parametro con indice de primer orden "
            "nulo tiene indice total de 0.244: influye solo a traves de interacciones.",
            evidence="indices de Sobol con N = 65 536: S3 = -0.0003, ST3 = 0.2437",
            recommendation="documentar la interaccion y anadirla a las pruebas de estres",
        )
    )
    r.add(
        Finding(
            "vigilancia_continua",
            Severity.HIGH,
            "no hay vigilancia de deriva de la relacion",
            "El seguimiento actual solo calcula PSI de las entradas, que es ciego a "
            "un cambio en la relacion entre entrada y salida.",
            evidence="caso construido: PSI = 0.000000 con el coeficiente invertido",
            recommendation="anadir seguimiento de coeficientes con revalidacion trimestral",
        )
    )
    for s in ("solidez_conceptual", "calidad_de_datos", "replica_independiente",
              "benchmarking", "analisis_de_resultados", "gobierno"):
        r.add(Finding(s, Severity.INFORMATIONAL, f"cobertura de {s}", "evaluado sin hallazgos"))

    print(f"Tras registrar la evidencia: {r.conclusion}")
    print(f"  hallazgos bloqueantes: {len(r.blocking_findings)}")
    print(f"  conteo por severidad : {r.severity_counts()}")
    print("\n--- primeras lineas del informe en Markdown ---")
    print("\n".join(r.to_markdown().splitlines()[:26]))


def main() -> None:
    demo_sobol()
    demo_morris()
    demo_estabilidad()
    demo_benchmarking()
    demo_informe()


if __name__ == "__main__":
    main()
