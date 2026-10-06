"""Servidor Flask: repositorio de modelos, comparación y clasificación por punto."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory)

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,150}$")
META_KEYS = {"id", "dataset", "modelo", "escalador", "reductor", "n_clases",
             "metricas", "matriz_confusion", "rangos_irradiancia"}
METRIC_KEYS = {"accuracy", "f1_macro", "mcc", "auc"}
CSV_COLS = ["latitude", "longitude", "value", "clase_real", "clase_pred"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB por subida

_cache = {}  # model_id -> (mtime, datos numpy)


# ----------------------------------------------------------------- utilidades
def paths(mid):
    return (MODELS_DIR / f"{mid}.json",
            MODELS_DIR / f"{mid}_predicciones.csv",
            MODELS_DIR / f"{mid}.joblib")


def etiquetas(n):
    return ["Baja", "Media", "Alta"] if n == 3 else [f"Clase {i}" for i in range(n)]


def read_meta(mid):
    pj, pc, pm = paths(mid)
    if not (ID_RE.match(mid) and pj.exists() and pc.exists()):
        return None
    meta = json.loads(pj.read_text(encoding="utf-8"))
    meta["etiquetas"] = etiquetas(int(meta["n_clases"]))
    meta["tiene_joblib"] = pm.exists()
    return meta


def list_models():
    out = []
    for p in sorted(MODELS_DIR.glob("*.json")):
        try:
            m = read_meta(p.stem)
            if m:
                out.append(m)
        except Exception:
            continue
    out.sort(key=lambda m: (m["dataset"], m.get("rank", 99)))
    return out


def median_nn(xy):
    """Distancia mediana al vecino más cercano (escala de la malla de puntos)."""
    if len(xy) < 2:
        return 1.0
    if len(xy) > 3000:
        xy = xy[np.random.default_rng(0).choice(len(xy), 3000, replace=False)]
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def load_points(mid):
    pj, pc, _ = paths(mid)
    if not (ID_RE.match(mid) and pc.exists()):
        return None
    mt = pc.stat().st_mtime
    if mid in _cache and _cache[mid][0] == mt:
        return _cache[mid][1]
    df = pd.read_csv(pc)[CSV_COLS].dropna()
    d = {
        "lat": df.latitude.to_numpy(float), "lon": df.longitude.to_numpy(float),
        "val": df.value.to_numpy(float),
        "real": df.clase_real.to_numpy(int), "pred": df.clase_pred.to_numpy(int),
    }
    d["nn"] = median_nn(np.c_[d["lat"], d["lon"]])
    _cache[mid] = (mt, d)
    return d


# ----------------------------------------------------------------------- rutas
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def api_models():
    return jsonify(models=list_models())


@app.post("/api/models")
def api_upload():
    """Almacena un modelo: metadatos (.json) + predicciones (.csv) + pipeline (.joblib opcional)."""
    fj, fc, fm = (request.files.get(k) for k in ("meta", "predicciones", "joblib"))
    if not fj or not fc:
        return jsonify(error="Se requieren el .json de metadatos y el CSV de predicciones."), 400
    try:
        meta = json.load(fj.stream)
    except Exception:
        return jsonify(error="El .json no es válido."), 400
    faltan = META_KEYS - set(meta) if isinstance(meta, dict) else META_KEYS
    if faltan:
        return jsonify(error=f"Al .json le faltan campos: {sorted(faltan)}"), 400
    if not METRIC_KEYS <= set(meta["metricas"]):
        return jsonify(error="'metricas' debe incluir accuracy, f1_macro, mcc y auc."), 400
    mid = str(meta["id"])
    if not ID_RE.match(mid):
        return jsonify(error="El 'id' solo admite letras, números, '-' y '_'."), 400
    pj, pc, pm = paths(mid)
    if pj.exists() and request.form.get("overwrite") != "1":
        return jsonify(error=f"Ya existe un modelo con id '{mid}'."), 409
    try:
        df = pd.read_csv(fc.stream)
    except Exception:
        return jsonify(error="El CSV no se pudo leer."), 400
    if any(c not in df.columns for c in CSV_COLS):
        return jsonify(error=f"El CSV debe tener las columnas {CSV_COLS}."), 400
    df = df[CSV_COLS].dropna()
    if df.empty:
        return jsonify(error="El CSV no tiene filas válidas."), 400
    if fm and fm.filename and not fm.filename.lower().endswith(".joblib"):
        return jsonify(error="El pipeline debe ser un archivo .joblib."), 400

    # Los nombres en disco salen del id validado, nunca del nombre subido.
    pj.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    df.to_csv(pc, index=False)
    if fm and fm.filename:
        fm.save(pm)  # solo se guarda; el servidor NUNCA lo deserializa (pickle)
    _cache.pop(mid, None)
    return jsonify(ok=True, id=mid), 201


@app.delete("/api/models/<mid>")
def api_delete(mid):
    if not ID_RE.match(mid) or not paths(mid)[0].exists():
        abort(404)
    for p in paths(mid):
        p.unlink(missing_ok=True)
    _cache.pop(mid, None)
    return jsonify(ok=True)


@app.get("/api/models/<mid>/joblib")
def api_joblib(mid):
    if not ID_RE.match(mid) or not paths(mid)[2].exists():
        abort(404)
    return send_from_directory(MODELS_DIR, f"{mid}.joblib", as_attachment=True)


@app.get("/api/models/<mid>/points")
def api_points(mid):
    d = load_points(mid)
    if d is None:
        return jsonify(error="Modelo no encontrado."), 404
    pts = np.c_[d["lat"], d["lon"], d["val"], d["real"], d["pred"]]
    return jsonify(
        bounds=dict(lat_min=float(d["lat"].min()), lat_max=float(d["lat"].max()),
                    lon_min=float(d["lon"].min()), lon_max=float(d["lon"].max())),
        median_nn=d["nn"], points=pts.tolist())


@app.post("/api/clasificar")
def api_clasificar():
    """Devuelve la clasificación del punto de la malla más cercano a (lat, lon)."""
    body = request.get_json(silent=True) or {}
    mid = str(body.get("model_id", ""))
    try:
        lat, lon = float(body["lat"]), float(body["lon"])
        assert np.isfinite(lat) and np.isfinite(lon)
    except Exception:
        return jsonify(error="lat y lon deben ser números."), 400
    meta = read_meta(mid) if ID_RE.match(mid) else None
    d = load_points(mid) if meta else None
    if d is None:
        return jsonify(error="Modelo no encontrado."), 404

    dist = np.hypot(d["lat"] - lat, d["lon"] - lon)
    i = int(dist.argmin())
    cp, cr = int(d["pred"][i]), int(d["real"][i])
    et = meta["etiquetas"]
    rangos = meta["rangos_irradiancia"]
    return jsonify(
        model_id=mid, clase_pred=cp, etiqueta_pred=et[cp] if cp < len(et) else str(cp),
        rango_pred=rangos.get(str(cp)), clase_real=cr,
        etiqueta_real=et[cr] if cr < len(et) else str(cr),
        valor=float(d["val"][i]), acierto=cp == cr,
        punto=dict(lat=float(d["lat"][i]), lon=float(d["lon"][i])),
        distancia=float(dist[i]), fuera_de_cobertura=bool(dist[i] > 2.5 * d["nn"]))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
