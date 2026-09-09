# Deprecated

Este directorio contiene experimentos y salidas históricas que se conservan
para trazabilidad, pero no forman parte de la ruta activa reutilizable.

## Scripts

| Archivo | Motivo |
|---|---|
| `scripts/run_stroke_general_ref03.py` | Benchmark tipo ARAI/intermedio. Ya no es dependencia del pipeline activo; la carga de datos corregida vive en `../stroke_data_adapter.py`. |
| `scripts/arai03_like_stroke.py` | Benchmark standalone inicial. Reemplazado por el pipeline constrained. |
| `scripts/run_stroke_03.py` | Estrategia original con FLAML/notebook. No es la ruta activa. |
| `scripts/report_stroke_general_results.py` | Utilidad antigua de reporte. Los resultados actuales se documentan desde `results.json` y el dashboard. |
| `scripts/monitor_stroke_general_cyberpunk.py` | Monitor terminal antiguo. Reemplazado por `../dashboard_template.html` + JSON del run. |

## Runs

| Directorio | Motivo |
|---|---|
| `runs/stroke_03_runs/` | Runs del enfoque original; incluye errores por dependencias y no representa el resultado actual. |
| `runs/stroke_03_ref_runs/` | Runs intermedios tipo ARAI; quedaron incompletos o fueron superados por el constrained run. |
| `constrained_backups/general_backup_20260811_101418/` | Snapshot del constrained run antes de la corrección de cohorte. Conservar solo para comparación histórica. |

## Regla

No usar nada de `deprecated/` para nuevos resultados salvo que se esté
reproduciendo historia explícitamente. Para nuevos proyectos, partir de:

```text
../run_stroke_general_constrained.py
../stroke_data_adapter.py
../dashboard_template.html
../docs/PORTING.md
```
