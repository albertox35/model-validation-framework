"""Generacion del informe de validacion, mapeado al marco regulatorio.

Citar solo **SR 11-7** en 2026 es incompleto. SR 11-7 es guia conjunta de la
Reserva Federal y la OCC, publicada en 2011 y aplicable a Estados Unidos. El
conjunto relevante hoy incluye:

- **SR 11-7 / OCC 2011-12** (Estados Unidos). Los tres pilares: solidez
  conceptual, vigilancia continua y analisis de resultados.
- **PRA SS1/23** (Reino Unido), vigente desde mayo de 2024. Es el marco de
  gestion de riesgo de modelos **mas reciente y explicito** que existe, con cinco
  principios y requisitos de gobierno, inventario y niveles de riesgo. Es justo
  el que buscaria un revisor europeo, y el que se omite al citar solo SR 11-7.
- **Guia TRIM del BCE** y *ECB Guide to internal models* (Union Europea).
- **CUB de la CNBV** y requisitos de validacion independiente de Banxico
  (Mexico).

El informe se estructura por **seccion del marco**, no por tipo de prueba. La
diferencia es practica: un validador tiene que demostrar cobertura de cada
requisito, y un informe organizado por tecnica obliga a reconstruir ese mapeo a
mano.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

__all__ = ["Severity", "Finding", "ValidationReport", "FRAMEWORK_SECTIONS"]


FRAMEWORK_SECTIONS: dict[str, dict[str, str]] = {
    "solidez_conceptual": {
        "sr_11_7": "III.A Desarrollo, implementacion y uso del modelo",
        "ss1_23": "Principio 2.2 Diseno y desarrollo del modelo",
        "descripcion": (
            "La teoria del modelo es adecuada al uso previsto, los supuestos estan "
            "documentados y la eleccion de metodo esta justificada frente a alternativas."
        ),
    },
    "calidad_de_datos": {
        "sr_11_7": "III.A Calidad y relevancia de los datos",
        "ss1_23": "Principio 2.3 Datos del modelo",
        "descripcion": (
            "Los datos de desarrollo son representativos de la poblacion de uso y su "
            "calidad esta verificada."
        ),
    },
    "replica_independiente": {
        "sr_11_7": "III.B Validacion de la implementacion",
        "ss1_23": "Principio 3.2 Validacion independiente",
        "descripcion": (
            "El codigo implementa lo que la documentacion especifica, verificado por "
            "reimplementacion independiente."
        ),
    },
    "analisis_de_sensibilidad": {
        "sr_11_7": "III.B Pruebas de sensibilidad y de estres",
        "ss1_23": "Principio 3.3 Alcance de la validacion",
        "descripcion": (
            "Se cuantifica la contribucion de cada parametro a la varianza de la "
            "salida, incluidas las interacciones, mediante analisis de sensibilidad "
            "global (Sobol y Morris) y no solo derivadas locales."
        ),
    },
    "benchmarking": {
        "sr_11_7": "III.B Comparacion con alternativas",
        "ss1_23": "Principio 3.3 Modelos retadores",
        "descripcion": (
            "Se compara el desempenio contra al menos un modelo retador para "
            "justificar la complejidad del modelo en produccion."
        ),
    },
    "analisis_de_resultados": {
        "sr_11_7": "III.B Analisis de resultados y backtesting",
        "ss1_23": "Principio 3.4 Seguimiento del desempenio",
        "descripcion": (
            "Las predicciones se contrastan contra lo observado, con pruebas "
            "estadisticas de sesgo y calibracion."
        ),
    },
    "vigilancia_continua": {
        "sr_11_7": "III.C Vigilancia continua",
        "ss1_23": "Principio 3.4 Seguimiento y revalidacion",
        "descripcion": (
            "Se vigila la estabilidad poblacional de las entradas y la deriva de la "
            "relacion entre entrada y salida."
        ),
    },
    "gobierno": {
        "sr_11_7": "V Gobierno, politicas y controles",
        "ss1_23": "Principio 1 Identificacion del modelo y gestion del riesgo",
        "descripcion": (
            "Hay inventario de modelos, propietario identificado, nivel de riesgo "
            "asignado y periodicidad de revalidacion definida."
        ),
    },
}
"""Mapeo de secciones de validacion a los marcos regulatorios."""


class Severity(str, Enum):
    """Severidad de un hallazgo de validacion."""

    CRITICAL = "critico"
    """Impide el uso del modelo hasta su remediacion."""

    HIGH = "alto"
    """Requiere remediacion con plazo definido y limitacion de uso."""

    MEDIUM = "medio"
    """Requiere remediacion en el ciclo ordinario."""

    LOW = "bajo"
    """Observacion de mejora, sin limitacion de uso."""

    INFORMATIONAL = "informativo"
    """Sin accion requerida."""

    @property
    def blocks_use(self) -> bool:
        """``True`` si la severidad impide aprobar el uso del modelo."""
        return self in (Severity.CRITICAL, Severity.HIGH)


@dataclass(frozen=True)
class Finding:
    """Un hallazgo de validacion."""

    section: str
    severity: Severity
    title: str
    detail: str
    evidence: str = ""
    recommendation: str = ""

    def __post_init__(self) -> None:
        if self.section not in FRAMEWORK_SECTIONS:
            raise ValueError(
                f"seccion desconocida: {self.section!r}. Las validas son "
                f"{sorted(FRAMEWORK_SECTIONS)}"
            )

    def to_markdown(self) -> str:
        """Representacion en Markdown del hallazgo."""
        partes = [f"**[{self.severity.value.upper()}] {self.title}**", "", self.detail]
        if self.evidence:
            partes += ["", f"*Evidencia:* {self.evidence}"]
        if self.recommendation:
            partes += ["", f"*Recomendacion:* {self.recommendation}"]
        return "\n".join(partes)


@dataclass
class ValidationReport:
    """Informe de validacion estructurado por seccion del marco regulatorio."""

    model_name: str
    model_owner: str
    validator: str
    validation_date: date
    model_use: str
    risk_tier: str = "medio"
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        """Anade un hallazgo."""
        self.findings.append(finding)

    def findings_by_section(self) -> dict[str, list[Finding]]:
        """Hallazgos agrupados por seccion."""
        agrupados: dict[str, list[Finding]] = {s: [] for s in FRAMEWORK_SECTIONS}
        for f in self.findings:
            agrupados[f.section].append(f)
        return agrupados

    @property
    def sections_without_evidence(self) -> list[str]:
        """Secciones del marco sin ningun hallazgo registrado.

        Una seccion vacia **no** significa que el modelo la cumpla: significa que
        no se evaluo. Distinguir «sin hallazgos» de «sin evaluar» es la diferencia
        entre un informe utilizable y uno que da falsa seguridad, y es el defecto
        mas comun de los informes generados automaticamente.
        """
        return [s for s, lista in self.findings_by_section().items() if not lista]

    @property
    def blocking_findings(self) -> list[Finding]:
        """Hallazgos que impiden aprobar el uso del modelo."""
        return [f for f in self.findings if f.severity.blocks_use]

    @property
    def conclusion(self) -> str:
        """Conclusion de la validacion.

        No se puede aprobar un modelo con secciones sin evaluar, aunque no haya
        hallazgos: la falta de evidencia no es evidencia de adecuacion.
        """
        if self.blocking_findings:
            return "NO APTO PARA USO"
        if self.sections_without_evidence:
            return "APROBACION CONDICIONADA: hay secciones del marco sin evaluar"
        return "APTO PARA EL USO PREVISTO"

    def severity_counts(self) -> dict[str, int]:
        """Conteo de hallazgos por severidad."""
        conteo = {s.value: 0 for s in Severity}
        for f in self.findings:
            conteo[f.severity.value] += 1
        return conteo

    def to_markdown(self) -> str:
        """Informe completo en Markdown, mapeado a SR 11-7 y PRA SS1/23."""
        lineas = [
            f"# Informe de validacion independiente: {self.model_name}",
            "",
            f"- **Propietario del modelo**: {self.model_owner}",
            f"- **Validador**: {self.validator}",
            f"- **Fecha de validacion**: {self.validation_date.isoformat()}",
            f"- **Uso previsto**: {self.model_use}",
            f"- **Nivel de riesgo**: {self.risk_tier}",
            "",
            "## Conclusion",
            "",
            f"**{self.conclusion}**",
            "",
        ]

        conteo = self.severity_counts()
        lineas += [
            "### Resumen de hallazgos",
            "",
            "| Severidad | Numero |",
            "|---|---|",
        ]
        for severidad in Severity:
            lineas.append(f"| {severidad.value} | {conteo[severidad.value]} |")
        lineas.append("")

        if self.sections_without_evidence:
            lineas += [
                "### Secciones sin evaluar",
                "",
                "Las siguientes secciones del marco no tienen evidencia registrada. "
                "La ausencia de hallazgos **no** equivale a cumplimiento:",
                "",
            ]
            for s in self.sections_without_evidence:
                lineas.append(f"- `{s}`")
            lineas.append("")

        lineas += ["## Hallazgos por seccion del marco", ""]
        for seccion, lista in self.findings_by_section().items():
            meta = FRAMEWORK_SECTIONS[seccion]
            lineas += [
                f"### {seccion.replace('_', ' ').capitalize()}",
                "",
                f"- **SR 11-7**: {meta['sr_11_7']}",
                f"- **PRA SS1/23**: {meta['ss1_23']}",
                "",
                meta["descripcion"],
                "",
            ]
            if not lista:
                lineas += ["> Sin evidencia registrada en esta validacion.", ""]
                continue
            for f in lista:
                lineas += [f.to_markdown(), ""]

        lineas += [
            "## Marco regulatorio de referencia",
            "",
            "- Federal Reserve SR 11-7 / OCC 2011-12, *Supervisory Guidance on Model "
            "Risk Management* (2011).",
            "- Bank of England PRA SS1/23, *Model risk management principles for banks* "
            "(vigente desde mayo de 2024).",
            "- BCE, *Guide to internal models* y guia TRIM.",
            "- CNBV, Circular Unica de Bancos, en materia de modelos internos y "
            "validacion independiente.",
            "",
        ]
        return "\n".join(lineas)
