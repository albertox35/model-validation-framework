"""Validacion del framework de validacion.

La prueba central contrasta los indices de Sobol contra los **valores analiticos**
de la funcion de Ishigami. Es el banco de pruebas estandar del area y tiene una
propiedad que lo hace especialmente exigente: un parametro con indice de primer
orden exactamente cero e indice total positivo.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

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
    ks_drift_test,
    morris_screening,
    population_stability_index,
    relationship_drift,
    replicate,
    sobol_indices,
)

BOUNDS = np.array([[-np.pi, np.pi]] * 3)


# ------------------------------------------------------------------ Sobol


def test_los_indices_analiticos_de_ishigami_son_los_publicados():
    """Valores de referencia de la literatura: S = (0.3139, 0.4424, 0) y
    S_T = (0.5576, 0.4424, 0.2437)."""
    an = ishigami_analytic_indices()
    np.testing.assert_allclose(an["first_order"], [0.3139, 0.4424, 0.0], atol=1e-4)
    np.testing.assert_allclose(an["total_order"], [0.5576, 0.4424, 0.2437], atol=1e-4)
    assert float(an["variance"]) == pytest.approx(13.8446, abs=1e-3)


def test_sobol_reproduce_los_indices_analiticos():
    an = ishigami_analytic_indices()
    res = sobol_indices(ishigami, BOUNDS, n_samples=32_768, seed=7)
    np.testing.assert_allclose(res.first_order, an["first_order"], atol=2e-3)
    np.testing.assert_allclose(res.total_order, an["total_order"], atol=2e-3)


def test_sobol_detecta_el_parametro_que_solo_actua_por_interaccion():
    """El caso que distingue un estimador correcto de uno aproximado.

    ``x3`` no aparece solo en la funcion de Ishigami: solo multiplica a
    ``sin x1``. Su indice de primer orden es exactamente cero y su indice total
    es 0.2437. Un estimador mal implementado acierta ``x1`` y ``x2``, que son
    grandes, y falla justo en este contraste.
    """
    res = sobol_indices(ishigami, BOUNDS, n_samples=32_768, seed=7)
    assert abs(res.first_order[2]) < 3.0 * res.first_order_stderr[2] + 1e-3
    assert res.total_order[2] == pytest.approx(0.2437, abs=2e-3)
    # Toda la influencia de x3 pasa por interaccion.
    assert res.interaction_strength[2] == pytest.approx(res.total_order[2], abs=2e-3)
    # Y x2 es puramente aditivo: no interactua con nada.
    assert res.interaction_strength[1] == pytest.approx(0.0, abs=2e-3)


def test_la_suma_de_los_indices_de_primer_orden_revela_las_interacciones():
    """Vale 1 solo si el modelo es aditivo. Ishigami no lo es."""
    res = sobol_indices(ishigami, BOUNDS, n_samples=16_384, seed=7)
    assert res.sum_first_order == pytest.approx(0.3139 + 0.4424, abs=5e-3)
    assert res.sum_first_order < 0.95


def test_el_error_de_sobol_decrece_al_aumentar_la_muestra():
    an = ishigami_analytic_indices()
    errores = []
    for n in (1_024, 8_192, 65_536):
        res = sobol_indices(ishigami, BOUNDS, n_samples=n, seed=7)
        errores.append(float(np.max(np.abs(res.total_order - an["total_order"]))))
    assert errores[-1] < errores[0], errores


def test_un_modelo_aditivo_tiene_indices_de_primer_orden_que_suman_uno():
    """Caso de control con interacciones nulas por construccion."""

    def aditivo(x: np.ndarray) -> np.ndarray:
        return 3.0 * x[:, 0] + 1.0 * x[:, 1] + 0.0 * x[:, 2]

    res = sobol_indices(aditivo, np.array([[0.0, 1.0]] * 3), n_samples=16_384, seed=3)
    assert res.sum_first_order == pytest.approx(1.0, abs=1e-3)
    np.testing.assert_allclose(res.interaction_strength, 0.0, atol=2e-3)
    # Y los indices reflejan la varianza aportada: 9:1:0.
    assert res.first_order[0] == pytest.approx(0.9, abs=5e-3)
    assert res.first_order[1] == pytest.approx(0.1, abs=5e-3)
    assert res.total_order[2] == pytest.approx(0.0, abs=2e-3)


def test_sobol_exige_potencia_de_dos():
    with pytest.raises(ValueError, match="no es potencia de dos"):
        sobol_indices(ishigami, BOUNDS, n_samples=1000)


def test_sobol_rechaza_un_modelo_constante():
    with pytest.raises(ValueError, match="no tiene varianza"):
        sobol_indices(lambda x: np.ones(x.shape[0]), BOUNDS, n_samples=1024)


def test_sobol_detecta_un_modelo_no_vectorizado():
    with pytest.raises(ValueError, match="vectorizado"):
        sobol_indices(lambda x: np.array([1.0]), BOUNDS, n_samples=1024)


def test_sobol_valida_las_cotas():
    with pytest.raises(ValueError, match="cota superior"):
        sobol_indices(ishigami, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), n_samples=1024)


def test_el_ranking_ordena_por_indice_total():
    res = sobol_indices(ishigami, BOUNDS, n_samples=8_192, seed=7, parameter_names=("a", "b", "c"))
    ranking = res.ranking()
    assert [n for n, _, _ in ranking] == ["a", "b", "c"]
    assert ranking[0][2] >= ranking[1][2] >= ranking[2][2]


# ------------------------------------------------------------------ Morris


def test_morris_identifica_un_parametro_irrelevante():
    """Cribado: el parametro sin efecto debe salir con influencia nula."""

    def modelo(x: np.ndarray) -> np.ndarray:
        return 5.0 * x[:, 0] + 0.5 * x[:, 1] + 0.0 * x[:, 2]

    res = morris_screening(
        modelo,
        np.array([[0.0, 1.0]] * 3),
        n_trajectories=200,
        seed=3,
        parameter_names=("fuerte", "debil", "nulo"),
    )
    assert res.mu_star[2] == pytest.approx(0.0, abs=1e-10)
    assert "nulo" in res.screen(threshold_fraction=0.05)
    assert "fuerte" not in res.screen(threshold_fraction=0.05)
    assert res.mu_star[0] > res.mu_star[1] > res.mu_star[2]


def test_morris_distingue_un_efecto_lineal_de_uno_no_lineal():
    """La desviacion de los efectos elementales delata la no linealidad.

    Un parametro con efecto lineal tiene el mismo efecto elemental en todo el
    dominio, asi que ``sigma = 0``. Uno con efecto no lineal o con interaccion
    tiene ``sigma`` grande. Es la informacion que ``mu_star`` sola no da.
    """

    def modelo(x: np.ndarray) -> np.ndarray:
        return 2.0 * x[:, 0] + np.sin(10.0 * x[:, 1])

    res = morris_screening(modelo, np.array([[0.0, 1.0]] * 2), n_trajectories=300, seed=5)
    # Efecto perfectamente lineal: el efecto elemental vale 2 en todo el dominio.
    assert res.sigma[0] == pytest.approx(0.0, abs=1e-8)
    assert res.mu_star[0] == pytest.approx(2.0, abs=1e-8)
    # Efecto no lineal: la dispersion es comparable a la magnitud del efecto, que
    # es la forma correcta de leerla. Un umbral absoluto depende de la escala del
    # modelo y no significa nada por si solo.
    assert res.sigma[1] > 0.3 * res.mu_star[1]
    # Y la razon entre ambas dispersiones es de muchos ordenes de magnitud.
    assert res.sigma[1] > 1e6 * max(res.sigma[0], 1e-300)


def test_morris_cuesta_mucho_menos_que_sobol():
    """La razon de cribar antes de cuantificar."""
    m = morris_screening(ishigami, BOUNDS, n_trajectories=100, seed=3)
    s = sobol_indices(ishigami, BOUNDS, n_samples=8_192, seed=3)
    assert m.n_evaluations == 100 * 4
    assert s.n_evaluations == 8_192 * 5
    assert m.n_evaluations < s.n_evaluations / 50


def test_morris_valida_sus_parametros():
    with pytest.raises(ValueError, match="n_levels"):
        morris_screening(ishigami, BOUNDS, n_levels=7)
    with pytest.raises(ValueError, match="al menos 2 trayectorias"):
        morris_screening(ishigami, BOUNDS, n_trajectories=1)


# ------------------------------------------------------------- estabilidad


def test_el_psi_de_dos_muestras_de_la_misma_poblacion_es_casi_nulo():
    rng = np.random.default_rng(1)
    psi = population_stability_index(rng.normal(size=20_000), rng.normal(size=20_000))
    assert psi < 0.01


def test_el_psi_crece_con_el_desplazamiento():
    rng = np.random.default_rng(1)
    base = rng.normal(size=20_000)
    valores = [
        population_stability_index(base, rng.normal(s, 1.0, 20_000)) for s in (0.1, 0.3, 0.6, 1.0)
    ]
    assert all(b > a for a, b in zip(valores, valores[1:])), valores


def test_el_psi_es_simetrico_con_tramos_fijos():
    """La divergencia de Jeffreys es simetrica; los tramos derivados no lo son.

    Con tramos fijos la simetria es exacta. Con tramos por percentiles de la
    referencia --el comportamiento por omision-- intercambiar los papeles cambia
    los tramos y la simetria solo es aproximada. Es la razon por la que, para
    comparar validaciones sucesivas, hay que fijar los tramos una vez.
    """
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 20_000)
    b = rng.normal(0.5, 1.2, 20_000)
    tramos = np.concatenate([[-np.inf], np.linspace(-3.0, 3.0, 11), [np.inf]])
    ab = population_stability_index(a, b, bin_edges=tramos)
    ba = population_stability_index(b, a, bin_edges=tramos)
    assert ab == pytest.approx(ba, rel=1e-12)


def test_los_umbrales_del_psi_se_reportan_como_convencion():
    rng = np.random.default_rng(1)
    base = rng.normal(size=20_000)
    informes = characteristic_stability(
        base.reshape(-1, 1), rng.normal(1.0, 1.0, 20_000).reshape(-1, 1), names=("x",)
    )
    assert informes[0].psi_verdict == "desplazamiento significativo"
    assert informes[0].ks_rejects()


def test_la_estabilidad_ordena_las_variables_por_psi():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(10_000, 3))
    cur = np.column_stack(
        [rng.normal(0.0, 1.0, 10_000), rng.normal(1.5, 1.0, 10_000), rng.normal(0.3, 1.0, 10_000)]
    )
    informes = characteristic_stability(ref, cur, names=("estable", "muy_movida", "poco_movida"))
    assert [i.name for i in informes] == ["muy_movida", "poco_movida", "estable"]


def test_kolmogorov_smirnov_no_rechaza_la_misma_poblacion():
    rng = np.random.default_rng(1)
    _, p = ks_drift_test(rng.normal(size=5_000), rng.normal(size=5_000))
    assert p > 0.05


def test_la_estabilidad_exige_las_mismas_variables():
    with pytest.raises(ValueError, match="deben coincidir"):
        characteristic_stability(np.zeros((100, 3)), np.zeros((100, 4)))


def test_la_deriva_de_relacion_detecta_lo_que_el_psi_no_ve():
    """La deriva que de verdad importa.

    Se invierte el signo de la relacion entre la primera variable y el objetivo,
    manteniendo **exactamente** la misma distribucion de las entradas. El PSI vale
    cero --no hay nada que detectar en las marginales-- y el modelo esta roto.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5_000, 2))
    y_ref = 2.0 * x[:, 0] + x[:, 1] + 0.1 * rng.normal(size=5_000)
    y_cur = -2.0 * x[:, 0] + x[:, 1] + 0.1 * rng.normal(size=5_000)

    for i in range(2):
        assert population_stability_index(x[:, i], x[:, i]) == pytest.approx(0.0, abs=1e-12)

    d = relationship_drift(x, y_ref, x, y_cur)
    assert d["max_coefficient_change"] > 1.5
    # El ajuste de cada periodo por separado sigue siendo bueno: el problema no
    # es que el fenomeno sea impredecible, es que cambio.
    assert d["r2_reference"] > 0.9 and d["r2_current"] > 0.9


def test_la_deriva_de_relacion_no_senala_nada_si_la_relacion_se_mantiene():
    rng = np.random.default_rng(11)
    x1 = rng.normal(size=(4_000, 2))
    x2 = rng.normal(size=(4_000, 2))
    y1 = 2.0 * x1[:, 0] + x1[:, 1] + 0.1 * rng.normal(size=4_000)
    y2 = 2.0 * x2[:, 0] + x2[:, 1] + 0.1 * rng.normal(size=4_000)
    d = relationship_drift(x1, y1, x2, y2)
    assert d["max_coefficient_change"] < 0.2
    assert abs(d["r2_degradation"]) < 0.05


# ------------------------------------------------------------ benchmarking


def test_la_replica_aprueba_una_implementacion_identica():
    a = np.linspace(1.0, 100.0, 500)
    res = replicate(a, a.copy())
    assert res.passed
    assert res.max_relative_difference == pytest.approx(0.0)


def test_la_replica_usa_diferencia_relativa_no_absoluta():
    """Una tolerancia absoluta es inutilizable si la salida recorre ordenes de magnitud."""
    a = np.array([1e-6, 1.0, 1e6])
    b = a * 1.001  # 0.1 % de error en los tres
    res = replicate(a, b, tolerance=1e-6)
    assert not res.passed
    assert res.n_exceeding == 3
    assert res.max_relative_difference == pytest.approx(0.001, rel=1e-6)


def test_la_replica_valida_las_formas():
    with pytest.raises(ValueError, match="misma forma"):
        replicate(np.zeros(10), np.zeros(5))


def test_el_retador_simple_puede_invalidar_la_complejidad():
    """La conclusion de validacion mas incomoda y mas valiosa.

    Si un retador simple empata con el modelo en produccion, la complejidad del
    segundo no esta justificada por su desempenio.
    """
    rng = np.random.default_rng(3)
    y = rng.normal(size=1_000)
    ruido = rng.normal(size=1_000)
    # El MISMO ruido escalado: la mejora es deterministica (2 %) y no depende de
    # dos realizaciones independientes que podrian diferir por azar.
    champion = y + 0.50 * ruido
    challenger = y + 0.51 * ruido
    comp = compare_challenger(y, champion, challenger)
    assert comp.improvement == pytest.approx(1.0 - 0.50 / 0.51, abs=1e-9)
    assert abs(comp.improvement) < 0.05
    assert not comp.complexity_justified()


def test_un_modelo_claramente_mejor_justifica_su_complejidad():
    rng = np.random.default_rng(3)
    y = rng.normal(size=1_000)
    champion = y + 0.2 * rng.normal(size=1_000)
    challenger = y + 1.0 * rng.normal(size=1_000)
    comp = compare_challenger(y, champion, challenger)
    assert comp.improvement > 0.5
    assert comp.complexity_justified()


def test_el_analisis_de_resultados_detecta_un_sesgo_significativo():
    rng = np.random.default_rng(5)
    y = rng.normal(size=5_000)
    sesgado = y + 0.1  # sesgo pequenio pero sistematico
    res = analyse_outcomes(y, sesgado)
    assert res.bias == pytest.approx(0.1, abs=0.01)
    assert res.bias_is_significant


def test_un_sesgo_pequenio_con_pocas_observaciones_no_es_significativo():
    """Por eso se reporta el estadistico t y no solo la media del error.

    El mismo sesgo puede ser significativo o ruido segun el tamanio de muestra.
    """
    rng = np.random.default_rng(5)
    # El error tiene que tener dispersion: un desplazamiento constante exacto es
    # infinitamente significativo con cualquier tamanio de muestra, porque su
    # error estandar es cero.
    sesgo, dispersion = 0.1, 0.5
    pocas = rng.normal(size=20)
    muchas = rng.normal(size=5_000)
    chica = analyse_outcomes(pocas, pocas + sesgo + dispersion * rng.normal(size=20))
    grande = analyse_outcomes(muchas, muchas + sesgo + dispersion * rng.normal(size=5_000))

    # El mismo sesgo: ruido con 20 observaciones, significativo con 5 000.
    assert not chica.bias_is_significant
    assert grande.bias_is_significant
    assert abs(grande.bias_t_statistic) > abs(chica.bias_t_statistic)


def test_el_analisis_de_resultados_exige_observaciones_suficientes():
    with pytest.raises(ValueError, match="al menos 3"):
        analyse_outcomes(np.zeros(2), np.zeros(2))


# ----------------------------------------------------------------- informe


def test_un_informe_sin_evidencia_no_aprueba_el_modelo():
    """La ausencia de hallazgos no es evidencia de adecuacion.

    Es el defecto mas comun de los informes generados automaticamente: una
    seccion vacia se lee como seccion cumplida. Aqui se distingue de forma
    explicita.
    """
    r = ValidationReport(
        model_name="motor-cva",
        model_owner="Tesoreria",
        validator="Riesgos",
        validation_date=date(2026, 11, 9),
        model_use="calculo de CVA para reporte regulatorio",
    )
    assert len(r.sections_without_evidence) == len(FRAMEWORK_SECTIONS)
    assert "sin evaluar" in r.conclusion
    assert not r.blocking_findings


def test_un_informe_con_todas_las_secciones_evaluadas_aprueba():
    r = ValidationReport("m", "o", "v", date(2026, 11, 9), "uso")
    for s in FRAMEWORK_SECTIONS:
        r.add(Finding(s, Severity.INFORMATIONAL, f"cobertura de {s}", "evaluado"))
    assert not r.sections_without_evidence
    assert r.conclusion == "APTO PARA EL USO PREVISTO"


def test_un_hallazgo_critico_bloquea_el_uso():
    r = ValidationReport("m", "o", "v", date(2026, 11, 9), "uso")
    for s in FRAMEWORK_SECTIONS:
        r.add(Finding(s, Severity.INFORMATIONAL, "ok", "evaluado"))
    r.add(Finding("replica_independiente", Severity.CRITICAL, "discrepancia", "3 % de diferencia"))
    assert r.conclusion == "NO APTO PARA USO"
    assert len(r.blocking_findings) == 1


@pytest.mark.parametrize(
    ("severidad", "bloquea"),
    [
        (Severity.CRITICAL, True),
        (Severity.HIGH, True),
        (Severity.MEDIUM, False),
        (Severity.LOW, False),
        (Severity.INFORMATIONAL, False),
    ],
)
def test_que_severidades_bloquean_el_uso(severidad, bloquea):
    assert severidad.blocks_use is bloquea


def test_el_informe_rechaza_secciones_desconocidas():
    with pytest.raises(ValueError, match="seccion desconocida"):
        Finding("inventada", Severity.LOW, "t", "d")


def test_cada_seccion_mapea_a_sr_11_7_y_a_ss1_23():
    """El mapeo al marco es el valor del informe, no un adorno."""
    for seccion, meta in FRAMEWORK_SECTIONS.items():
        assert meta["sr_11_7"], seccion
        assert meta["ss1_23"], seccion
        assert meta["descripcion"], seccion


def test_el_informe_en_markdown_incluye_el_mapeo_y_las_secciones_vacias():
    r = ValidationReport("motor-cva", "Tesoreria", "Riesgos", date(2026, 11, 9), "CVA")
    r.add(
        Finding(
            "analisis_de_sensibilidad",
            Severity.MEDIUM,
            "interaccion no documentada",
            "x3 solo actua por interaccion",
            evidence="indices de Sobol: S3 = 0.000, ST3 = 0.244",
            recommendation="documentar la interaccion en la especificacion",
        )
    )
    md = r.to_markdown()
    assert "SR 11-7" in md
    assert "PRA SS1/23" in md
    assert "Secciones sin evaluar" in md
    assert "indices de Sobol" in md
    assert "APROBACION CONDICIONADA" in md


def test_el_conteo_por_severidad_cubre_todas_las_categorias():
    r = ValidationReport("m", "o", "v", date(2026, 11, 9), "uso")
    r.add(Finding("gobierno", Severity.HIGH, "t", "d"))
    r.add(Finding("gobierno", Severity.LOW, "t", "d"))
    conteo = r.severity_counts()
    assert set(conteo) == {s.value for s in Severity}
    assert conteo["alto"] == 1 and conteo["bajo"] == 1 and conteo["critico"] == 0
