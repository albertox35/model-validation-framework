# model-validation-framework

Validación independiente de modelos cuantitativos: análisis de sensibilidad
**global** (Sobol y Morris), estabilidad poblacional, réplica independiente,
modelo retador, análisis de resultados, e informe mapeado **sección por sección**
a SR 11-7 y PRA SS1/23.

```python
from modelval import sobol_indices, ishigami, ishigami_analytic_indices

r = sobol_indices(ishigami, bounds, n_samples=65_536)
print(r.first_order)           # [ 0.3136  0.4424 -0.0003]
print(r.total_order)           # [ 0.5576  0.4424  0.2437]
print(r.interaction_strength)  # [ 0.244   0.      0.244 ]
# x3 tiene índice de primer orden cero y total 0.244: solo actúa por interacción
```

---

## El marco citado estaba incompleto

Citar solo **SR 11-7** en 2026 deja fuera lo más relevante. SR 11-7 es guía
conjunta de la Reserva Federal y la OCC, de **2011**, aplicable a Estados Unidos.
El conjunto vigente incluye:

| Marco | Jurisdicción | Nota |
|---|---|---|
| SR 11-7 / OCC 2011-12 | EE. UU. | 2011. Los tres pilares clásicos |
| **PRA SS1/23** | Reino Unido | **Vigente desde mayo de 2024.** El marco de gestión de riesgo de modelos más reciente y explícito: cinco principios, inventario, niveles de riesgo |
| Guía TRIM y *ECB Guide to internal models* | Unión Europea | Modelos internos |
| CUB de la CNBV, Banxico | México | Validación independiente |

**PRA SS1/23 es justo el que buscaría un revisor europeo**, y el que se omite al
citar solo SR 11-7. El informe de este repositorio mapea cada sección a los dos.

---

## Validación: Sobol contra los valores analíticos de Ishigami

La función de Ishigami es el banco de pruebas estándar del área, y no por
tradición: tiene un parámetro que **no aparece solo** en la fórmula, solo
multiplicando a otro término. Su índice de primer orden es **exactamente cero** y
su índice total es **positivo**. Un estimador mal implementado acierta los dos
índices grandes y falla justo en ese contraste, que es el que distingue un
análisis global de uno local.

| N | S₁ | S₂ | S₃ | ST₁ | ST₂ | ST₃ |
|---|---|---|---|---|---|---|
| 1 024 | 0.3079 | 0.4444 | −0.0036 | 0.5552 | 0.4413 | 0.2437 |
| 4 096 | 0.3118 | 0.4428 | −0.0025 | 0.5569 | 0.4426 | 0.2439 |
| 16 384 | 0.3135 | 0.4424 | −0.0004 | 0.5575 | 0.4424 | 0.2436 |
| 65 536 | 0.3136 | 0.4424 | **−0.0003** | 0.5576 | 0.4424 | **0.2437** |
| **exacto** | **0.3139** | **0.4424** | **0.0000** | **0.5576** | **0.4424** | **0.2437** |

Error máximo en los índices totales: **9.2e-06**.

Se usan **dos estimadores distintos a propósito** (Saltelli et al., 2010): el de
Saltelli para primer orden y el de **Jansen** para el orden total. El de Jansen es
claramente superior cuando el índice es pequeño: el estimador clásico resta dos
cantidades casi iguales y el error relativo se dispara, mientras el de Jansen es
una suma de cuadrados y no sufre cancelación.

---

## Morris: cribar antes de cuantificar

| Parámetro | μ* | σ | Lectura |
|---|---|---|---|
| lineal | 5.0000 | **0.0000** | efecto lineal |
| no lineal | 0.7032 | 0.7300 | no lineal o con interacción |
| nulo | 0.0000 | 0.0000 | irrelevante: puede fijarse |

**σ = 0 significa efecto idéntico en todo el dominio.** Es información que μ* sola
no da, y la razón de reportar las dos.

Costo: **1 200 evaluaciones frente a 40 960** de Sobol en el mismo problema, 34×
menos. El flujo correcto es cribar con Morris y cuantificar con Sobol solo lo que
sobrevive.

Un detalle de implementación que importa: cuando un salto saldría del cubo
unidad, se **invierte el signo**, no se recorta. Recortar produce saltos de
longitud distinta a δ y, en el borde, saltos de longitud cero cuyo efecto
elemental no está definido. Esos ceros espurios contaminan σ, que es justamente
la cantidad que mide la no linealidad: un parámetro con efecto perfectamente
lineal pasaría a tener σ > 0. Hay una prueba que exige σ = 0 exacto para un
efecto lineal.

---

## La deriva que el PSI no puede ver

Se invierte el signo de la relación entre una variable y el objetivo, manteniendo
**exactamente** la misma distribución de las entradas:

| Diagnóstico | Valor |
|---|---|
| PSI de x₁ | **0.000000** |
| PSI de x₂ | **0.000000** |
| Cambio máximo de coeficientes | **1.9985** |
| R² referencia / actual | 0.9980 / 0.9980 |

El PSI vale cero y el modelo está roto. Cada periodo se ajusta bien por separado
—el fenómeno no es impredecible, **cambió**— así que un modelo entrenado en el
primero fallaría por completo en el segundo.

Es la deriva que más importa y la que un seguimiento basado solo en PSI no
detecta nunca.

---

## Réplica, retador y análisis de resultados son tres cosas distintas

Suelen confundirse en una sola, y cada una responde a una pregunta diferente:

**Réplica independiente.** ¿El código implementa lo que dice la documentación? Se
mide diferencia **relativa**, no absoluta: una tolerancia absoluta es
inutilizable cuando la salida recorre varios órdenes de magnitud, que es el caso
de un precio o una exposición.

**Modelo retador.** ¿La elección de modelo está justificada?

| Comparación | RMSE producción | RMSE retador | Mejora | Veredicto |
|---|---|---|---|---|
| retador casi idéntico | 0.505135 | 0.515237 | +2.0 % | **complejidad NO justificada** |
| retador claramente peor | 0.202054 | 1.010269 | +80.0 % | complejidad justificada |

Si un retador simple empata con el modelo en producción, la complejidad del
segundo **no está justificada por su desempeño**. Es la conclusión de validación
más incómoda y la que más valor aporta.

**Análisis de resultados.** ¿Las predicciones se cumplieron?

| n | Sesgo | t | Veredicto |
|---|---|---|---|
| 20 | +0.0254 | +0.23 | no significativo |
| 5 000 | +0.1080 | **+15.45** | **significativo** |

El mismo sesgo es ruido con 20 observaciones y significativo con 5 000. Por eso
se reporta el **estadístico t**, no solo la media del error: reportar solo la
media invita a la conclusión equivocada en los dos sentidos.

---

## Una sección sin evaluar no es una sección cumplida

Es el defecto más común de los informes de validación generados
automáticamente: una sección vacía se lee como sección cumplida.

```
Informe recién abierto: APROBACIÓN CONDICIONADA: hay secciones del marco sin evaluar
  secciones del marco sin evidencia: 8 de 8
```

El informe distingue explícitamente «sin hallazgos» de «sin evaluar», y **no
aprueba** un modelo con secciones sin evidencia aunque no haya hallazgo alguno.
La falta de evidencia no es evidencia de adecuación.

Las ocho secciones, cada una mapeada a SR 11-7 y a SS1/23: solidez conceptual,
calidad de datos, réplica independiente, análisis de sensibilidad, benchmarking,
análisis de resultados, vigilancia continua y gobierno.

---

## Validación del propio framework

44 pruebas. Reproducible con `python examples/validate_model.py`.

| Propiedad verificada | Resultado |
|---|---|
| Sobol reproduce los índices analíticos de Ishigami | error ≤ 9.2e-06 en ST |
| Detecta el parámetro que solo actúa por interacción | S₃ ≈ 0, ST₃ = 0.2437 |
| Un modelo aditivo da Σ S_i = 1 e interacciones nulas | exacto a 1e-3 |
| Los índices reflejan la varianza aportada (9:1:0) | exacto a 5e-3 |
| El error de Sobol decrece con N | sí |
| Morris da σ = 0 exacto para un efecto lineal | sí (prueba del arreglo del salto) |
| Morris identifica el parámetro irrelevante | μ* = 0 exacto |
| PSI simétrico con tramos fijos | exacto a 1e-12 |
| PSI monótono en el desplazamiento | sí |
| La deriva de relación detecta lo que el PSI no ve | sí |
| Un informe sin evidencia no aprueba el modelo | sí |
| Un hallazgo crítico o alto bloquea el uso | sí |

---

## Qué no hace

- **No valida automáticamente los otros repositorios del conjunto.** El framework
  es agnóstico del modelo: toma una función vectorizada y unas cotas. Escribir los
  adaptadores concretos para cada motor es un trabajo de integración que no está
  hecho.
- **Sin métricas de clasificación.** No hay AUC, curva ROC, KS de puntuación,
  matriz de confusión ni curva de calibración, que son el núcleo de la validación
  de modelos de scoring. El enfoque aquí es de modelos de valoración y riesgo, con
  salida continua.
- **Sin pruebas de estrés ni escenarios inversos.** SR 11-7 y SS1/23 las exigen;
  el framework entrega la sensibilidad global, que es el insumo, no el escenario.
- **Sin gestión del inventario de modelos.** SS1/23 pide inventario con
  propietario, nivel de riesgo y periodicidad de revalidación. El informe tiene
  los campos; no hay base de datos detrás.
- **Sin interpretabilidad de modelos de aprendizaje automático** (SHAP, LIME,
  gráficos de dependencia parcial). Los índices de Sobol cumplen una función
  parecida y con mejor fundamento teórico, pero son más caros y necesitan poder
  reevaluar el modelo a voluntad.
- **Sobol exige poder reevaluar el modelo** `N(d+2)` veces. Para un motor que
  tarda minutos por evaluación, eso es inviable sin un metamodelo, que no está
  implementado.
- **El `relationship_drift` ajusta una regresión lineal.** Para un modelo no
  lineal detecta cambios en la relación promedio, no en su forma.

---

## Alcance y decisiones de diseño

**Sensibilidad global, no local.** Las derivadas parciales en un punto no sirven
para validar un modelo no lineal: el resultado depende del punto y no dice nada
sobre interacciones. El framework no ofrece sensibilidad local, a propósito.

**Los umbrales del PSI se reportan como convención.** Los valores 0.10 y 0.25 son
convención de la industria, **no** niveles de significancia: no hay ninguna
distribución nula detrás. Se exponen como guía junto a la prueba de
Kolmogorov-Smirnov, que sí tiene p-valor, con la advertencia de que la potencia
del KS crece con el tamaño de muestra y con millones de observaciones rechaza
desplazamientos irrelevantes.

**El PSI construye los tramos por percentiles de la referencia**, no
equiespaciados: con una variable asimétrica los tramos equiespaciados dejan
tramos vacíos y el índice se vuelve inestable. La contrapartida es que la simetría
`PSI(a,b) = PSI(b,a)` pasa a ser aproximada, así que para comparar validaciones
sucesivas hay que fijar `bin_edges` una vez. Está documentado y hay una prueba de
la simetría exacta con tramos fijos.

**El cambio relativo de coeficientes tiene un suelo en el denominador** del 1 %
del coeficiente mayor. Sin ese suelo, un coeficiente cuyo valor verdadero es cero
—típicamente la constante— se estima como un número diminuto distinto en cada
periodo, y el diagnóstico se dispararía por el coeficiente que menos importa. Es
el fallo clásico de medir cambios en términos relativos.

**El repositorio es autocontenido.** No depende de una librería núcleo compartida
con los demás del conjunto.

---

## Instalación

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
python examples/validate_model.py
```

Requiere Python 3.11 o superior, NumPy, SciPy y pandas.

---

## Referencias

- Federal Reserve SR 11-7 / OCC 2011-12 (2011). *Supervisory Guidance on Model
  Risk Management*.
- Bank of England, Prudential Regulation Authority (2023). *SS1/23 Model risk
  management principles for banks*. Vigente desde mayo de 2024.
- Banco Central Europeo (2019). *Guide to internal models*; proyecto TRIM.
- Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M. & Tarantola, S.
  (2010). *Variance based sensitivity analysis of model output: design and
  estimator for the total sensitivity index*. Computer Physics Communications
  181(2).
- Jansen, M. (1999). *Analysis of variance designs for model output*. Computer
  Physics Communications 117(1-2).
- Sobol, I. (2001). *Global sensitivity indices for nonlinear mathematical models
  and their Monte Carlo estimates*. Mathematics and Computers in Simulation
  55(1-3).
- Morris, M. (1991). *Factorial sampling plans for preliminary computational
  experiments*. Technometrics 33(2).
- Campolongo, F., Cariboni, J. & Saltelli, A. (2007). *An effective screening
  design for sensitivity analysis of large models*. Environmental Modelling &
  Software 22(10).
- Ishigami, T. & Homma, T. (1990). *An importance quantification technique in
  uncertainty analysis for computer models*. Proceedings of ISUMA.
- Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2ª ed. Wiley. Para la práctica
  del PSI.

## Licencia

MIT. Ver [LICENSE](LICENSE).
