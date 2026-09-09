# Portar el pipeline a otro proyecto

La forma limpia de reutilizar esta estrategia es tratar el pipeline como motor
y aislar lo específico del proyecto en un adaptador de datos.

## Archivos que hay que copiar

Mínimo:

```text
run_stroke_general_constrained.py
stroke_data_adapter.py
dashboard_template.html
requirements_stroke_challengers.txt
```

En el proyecto nuevo, se recomienda renombrarlos:

```text
run_<proyecto>_constrained.py
<proyecto>_data_adapter.py
dashboard_template.html
requirements_<proyecto>.txt
```

## Cambios obligatorios

1. Cambiar el import del adaptador.

```python
from <proyecto>_data_adapter import configure_environment
from <proyecto>_data_adapter import prepare_data
```

2. Reescribir `prepare_data()` para devolver `X, y, year`.

3. Revisar la lista de modelos en `build_stable_models()`:

- Mantener modelos compatibles con los datos y dependencias disponibles.
- Ajustar espacios de hiperparámetros si cambian tamaño muestral, prevalencia o coste computacional.
- Mantener `random_state=42` o fijar una semilla equivalente.

4. Revisar el objetivo clínico/operativo del umbral:

```python
primary = max_spec_at_recall_point(y_true, probabilities, 0.80)
```

Para otro proyecto, `0.80` debe justificarse. Si el requisito cambia, cambiar
ese valor y documentarlo.

## Contrato de `prepare_data()`

Debe cumplir:

- `X.index`, `y.index` y `year.index` alineados.
- `y` binaria `0/1`.
- `X` sin columnas target, sin columnas derivadas del desenlace y sin fugas.
- Variables categóricas codificadas antes de devolver `X`, o gestionadas por
  modelos/pipelines compatibles.
- Valores infinitos convertidos a missing.
- Tipos numéricos compatibles con sklearn.

Ejemplo mínimo:

```python
def prepare_data():
    df = load_project_table()
    y = df["target"].astype(int)
    year = df["year"] if "year" in df else None
    X = df.drop(columns=["target", "year"], errors="ignore")
    X = pd.get_dummies(X, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    return X, y, year
```

## Salida esperada del run

El pipeline crea o actualiza:

```text
<RUN_DIR>/
  dashboard.html
  split.npz
  state.json
  parallel_progress.json
  results.json
  training.log
  <modelo>.joblib
```

Si falta `dashboard.html`, el script lo copia desde
`experiments/dashboard_template.html`.

## Reanudar ejecuciones

Usar `--resume` para conservar:

- split congelado
- modelos ya completados
- historial HPO guardado
- estado del dashboard

Ejemplo:

```bash
python3 experiments/run_<proyecto>_constrained.py \
  --run-dir experiments/<proyecto>_constrained_runs/general \
  --resume
```

Si se cambia el dataset, target, filtros, columnas o criterio de umbral, no
reutilizar el `split.npz` anterior. Crear un `RUN_DIR` nuevo o mover el run
antiguo a deprecated.

## Checklist anti-leakage

Antes de lanzar:

- Confirmar que `prepare_data()` no incluye target ni variables posteriores al desenlace.
- Confirmar que filtros de cohorte se aplican antes del split.
- Confirmar que transformaciones aprendidas de datos están dentro de pipelines sklearn.
- Confirmar que no se mira test para escoger modelo, hiperparámetros o umbral.
- Confirmar que el dashboard solo lee estado y no alimenta decisiones manuales sobre test.

Después de lanzar:

- Revisar `state.json.status`.
- Revisar `results.json.winner`.
- Reportar métricas de `test_metrics`, indicando que el umbral viene de OOF train.
- Conservar `split.npz` junto a resultados para reproducibilidad.
