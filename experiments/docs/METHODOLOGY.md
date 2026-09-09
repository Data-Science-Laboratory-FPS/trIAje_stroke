# Metodología del pipeline constrained

El objetivo del pipeline es seleccionar modelo, hiperparámetros y umbral sin
usar el test hasta el final. El 20% de test queda como evaluación ciega: no
participa en ranking, HPO, selección de umbral ni decisiones manuales.

## Contrato de entrada

`run_stroke_general_constrained.py` espera que el adaptador de datos exponga:

```python
def configure_environment() -> None: ...
def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.Series | None]: ...
```

`prepare_data()` debe devolver:

| Salida | Requisito |
|---|---|
| `X` | `DataFrame` numérico, una fila por observación, sin target ni columnas de fuga. |
| `y` | Serie binaria `0/1`, alineada fila a fila con `X`. |
| `year` | Serie auxiliar alineada, usada como metadato. El pipeline actual no estratifica por año. |

La imputación y el escalado se hacen dentro de cada pipeline sklearn, no antes
de la validación. Esto evita que estadísticas de folds de validación entren en
el entrenamiento de cada fold.

## Fase 0: split congelado

El script crea un split estratificado:

```python
train_test_split(np.arange(len(y)), test_size=0.20, random_state=42, stratify=y)
```

Los índices se guardan en `split.npz`. Con `--resume`, si existe `split.npz`,
se reutiliza exactamente el mismo split.

Regla: el test no se usa hasta la fase final.

## Fase 1: screening de modelos

Sobre el 80% de train:

- CV: `StratifiedKFold(3, shuffle=True, random_state=42)`
- Métrica: `Average Precision` OOF
- Modelos candidatos:
  - `XGBoost`
  - `Random Forest`
  - `Logistic Regression`
  - `Elastic Net Logistic`
  - `Extra Trees`
  - `LightGBM`
  - `HistGradientBoosting`
- Salen adelante los `--survivors` mejores por AP, por defecto 4.

Esta fase reduce coste sin mirar test.

## Fase 2: HPO

Solo sobre los supervivientes:

- CV: `StratifiedKFold(5, shuffle=True, random_state=42)`
- Búsqueda: `ParameterSampler`
- Candidatos por modelo: `--hpo-candidates`, por defecto 12
- Métrica primaria: `Average Precision` OOF

Para cada modelo superviviente se guarda:

- historial de candidatos en `state.json`
- mejor combinación en `best_hpo`
- progreso activo para el dashboard

## Fase 3: selección de umbral

Con los mejores hiperparámetros de cada modelo:

- CV: `StratifiedKFold(10, shuffle=True, random_state=42)`
- Se generan probabilidades OOF en train.
- Se barre el umbral de probabilidad.
- El punto primario maximiza especificidad sujeto a `recall >= 0.80`.

Criterio:

```text
max specificity where recall >= 0.80
```

Si ningún umbral alcanza el recall mínimo, se elige el punto de mayor recall y,
como desempate, mayor especificidad y menos falsos positivos.

También se guardan puntos secundarios:

- `recall_0.80`
- `recall_0.85`
- `recall_0.90`
- `f1`
- `f2`

El umbral queda fijado aquí usando solo train.

## Fase 4: refit final

Para cada modelo superviviente completado:

1. Se clona el estimador con sus mejores hiperparámetros.
2. Se entrena sobre todo el 80% de train.
3. Se serializa como `.joblib`.

## Fase 5: evaluación ciega de test

El modelo refiteado predice probabilidades sobre el 20% test. Se aplica el
umbral fijado en OOF train y se calculan métricas de test:

- recall
- specificity
- precision / PPV
- NPV
- F1
- F2
- TP, TN, FP, FN
- Average Precision
- AUROC
- log loss
- Brier score

Estos son los números reportables.

## Ranking final

El ganador se elige entre modelos completados con esta prioridad:

```text
feasible recall >= 0.80
specificity OOF
Average Precision OOF
menos falsos positivos OOF
```

La comparación entre modelos usa OOF de train, no test. El test queda para
lectura final honesta del ganador y del resto de modelos completados.

## Archivos generados

| Archivo | Generado por | Función |
|---|---|---|
| `split.npz` | Fase 0 | Índices train/test congelados. |
| `state.json` | Todas las fases | Checkpoint reanudable y estado principal del dashboard. |
| `parallel_progress.json` | CV paralela | Heartbeats por worker/fold para el dashboard. |
| `parallel_progress.lock` | CV paralela | Lock de escritura concurrente. No se copia manualmente. |
| `dashboard.html` | Inicio del run | Copia de `dashboard_template.html` si no existía en el directorio del run. |
| `results.json` | Final | Resumen final ordenado, ganador y métricas completas. |
| `*.joblib` | Refit final | Modelos entrenados sobre todo train. |
| `training.log` | Toda la ejecución | Log de entrenamiento. |
| `runtime_status.json` | Watcher externo opcional | PID, CPU y memoria para el dashboard. |

## Propiedades anti-leakage

- El split test se congela antes de tomar decisiones.
- Los folds son estratificados y reproducibles.
- La imputación y el escalado están dentro de cada fold sklearn.
- La selección de modelo usa OOF train.
- La selección de hiperparámetros usa OOF train.
- La selección de umbral usa OOF train.
- El test solo se evalúa al final con modelo y umbral ya cerrados.
