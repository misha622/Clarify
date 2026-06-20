"""
Clarify Web UI вЂ” FastAPI СЃРµСЂРІРµСЂ.

Р—Р°РїСѓСЃРє:
    python -m src.api.server
    python -m src.api.server --port 8000 --lang ru

РћС‚РєСЂС‹С‚СЊ РІ Р±СЂР°СѓР·РµСЂРµ: http://localhost:8000
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, ".")

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import numpy as np
import xgboost as xgb
import yaml

from src.data.synthetic_generator import SyntheticGenerator
from src.explainers.shap_explainer import ShapExplainer
from src.rendering.template_renderer import TemplateRenderer
from src.ui.alert_card import AlertCardBuilder
from src.ui.confirm_flow import ConfirmFlow, WebhookConfig, FirewallCommand
from src.detectors.dga import DGADetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Clarify Autonomous Security Layer", version="0.2.0")
# === API Key Middleware ===
from starlette.middleware.base import BaseHTTPMiddleware
import os

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Простой API-key middleware. Установи CLARIFY_API_KEY в env. Без ключа — dev mode (пропускает)."""
    async def dispatch(self, request, call_next):
        if request.url.path in ("/api/health", "/", "/favicon.ico"):
            return await call_next(request)
        expected = os.environ.get("CLARIFY_API_KEY", "")
        if not expected:
            return await call_next(request)
        if request.headers.get("X-API-Key", "") != expected:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

app.add_middleware(APIKeyMiddleware)

# Р“Р»РѕР±Р°Р»СЊРЅС‹Рµ РєРѕРјРїРѕРЅРµРЅС‚С‹
COMPONENTS = {}
ALERT_CACHE: dict[str, dict] = {}
BLOCKED_IPS: set = set()
IGNORED_ALERTS: set = set()
CONFIRM_FLOW: Optional[ConfirmFlow] = None


def init_components(lang: str = "ru"):
    """РРЅРёС†РёР°Р»РёР·РёСЂСѓРµС‚ РІСЃРµ РєРѕРјРїРѕРЅРµРЅС‚С‹ Clarify."""
    global CONFIRM_FLOW

    with open("config/detectors.yaml", "r") as f:
        config = yaml.safe_load(f)

    components = {"config": config}

    # Beaconing
    bc_cfg = config["detectors"]["beaconing"]
    if Path(bc_cfg["model_path"]).exists():
        model = xgb.Booster()
        model.load_model(bc_cfg["model_path"])
        components["beaconing_model"] = model
        components["beaconing_threshold"] = bc_cfg["decision_threshold"]
        components["beaconing_explainer"] = ShapExplainer(
            model, bc_cfg["features"], top_n=3
        )
        logger.info("Beaconing Р·Р°РіСЂСѓР¶РµРЅ")

    # Brute-Force
    bf_cfg = config["detectors"].get("brute_force", {})
    if bf_cfg.get("model_path") and Path(bf_cfg["model_path"]).exists():
        model = xgb.Booster()
        model.load_model(bf_cfg["model_path"])
        components["brute_force_model"] = model
        components["brute_force_threshold"] = bf_cfg["decision_threshold"]
        components["brute_force_explainer"] = ShapExplainer(
            model, bf_cfg["features"], top_n=3
        )
        logger.info("Brute-Force Р·Р°РіСЂСѓР¶РµРЅ")

    # DGA
    dga_cfg = config["detectors"].get("dga", {})
    if dga_cfg.get("model_path") and Path(dga_cfg["model_path"]).exists():
        model = xgb.Booster()
        model.load_model(dga_cfg["model_path"])
        components["dga_model"] = model
        components["dga_threshold"] = dga_cfg["decision_threshold"]
        components["dga_explainer"] = ShapExplainer(
            model, dga_cfg.get("features", DGADetector.FEATURE_NAMES), top_n=3
        )
        logger.info("DGA Р·Р°РіСЂСѓР¶РµРЅ")

    # Р РµРЅРґРµСЂРµСЂ
    dict_path = f"config/feature_dictionary{'_en' if lang == 'en' else ''}.yaml"
    if not Path(dict_path).exists():
        dict_path = "config/feature_dictionary.yaml"
    components["renderer"] = TemplateRenderer(dictionary_path=dict_path)
    components["builder"] = AlertCardBuilder(template_renderer=components["renderer"])

    # Р“РµРЅРµСЂР°С‚РѕСЂ РґР»СЏ РґРµРјРѕ
    components["generator"] = SyntheticGenerator(seed=42)

    # Confirm-flow
    CONFIRM_FLOW = ConfirmFlow(webhook_config=None)

    return components


def generate_demo_alerts(n: int = 10, use_cache: bool = True) -> list[dict]:
    """Р“РµРЅРµСЂРёСЂСѓРµС‚ РґРµРјРѕ-Р°Р»РµСЂС‚С‹. РљРµС€РёСЂСѓРµС‚ РїРѕ ID РґР»СЏ РґРµС‚Р°Р»СЊРЅРѕРіРѕ РїСЂРѕСЃРјРѕС‚СЂР°."""
    global ALERT_CACHE

    if not use_cache or len(ALERT_CACHE) == 0:
        ALERT_CACHE = {}
        gen = COMPONENTS.get("generator", SyntheticGenerator(seed=42))
        builder: AlertCardBuilder = COMPONENTS["builder"]

        X, y = gen.generate_for_beaconing_training(
            mode="train", window_size_seconds=900,
            stride_seconds=300, min_events_per_window=8, num_hosts=50,
        )

        attack_indices = [i for i, label in enumerate(y) if label == 1]

        for i, idx in enumerate(attack_indices[:n]):
            sample = X[idx:idx + 1]

            if i % 3 == 0 and "beaconing_explainer" in COMPONENTS:
                shap = COMPONENTS["beaconing_explainer"].explain(
                    sample, "beaconing", context={"source_ip": f"45.33.32.{100 + i}"}
                )
                dmatrix = xgb.DMatrix(
                    sample,
                    feature_names=COMPONENTS["beaconing_explainer"].feature_names,
                )
                score = float(COMPONENTS["beaconing_model"].predict(dmatrix)[0])
                card = builder.build(
                    "beaconing", f"45.33.32.{100 + i}", "10.0.5.17",
                    score, COMPONENTS["beaconing_threshold"], shap, "beaconing",
                )
                ALERT_CACHE[card.alert_id] = card.to_dict()

            elif i % 3 == 1 and "brute_force_explainer" in COMPONENTS:
                bf_sample = np.random.RandomState(i).uniform(0, 1, (1, 5))
                bf_sample[0, 0] = 55.0
                bf_sample[0, 1] = 15.0
                bf_sample[0, 2] = 5.0

                shap = COMPONENTS["brute_force_explainer"].explain(
                    bf_sample, "brute_force",
                    context={"source_ip": f"203.0.113.{200 + i}"}
                )
                dmatrix = xgb.DMatrix(
                    bf_sample,
                    feature_names=COMPONENTS["brute_force_explainer"].feature_names,
                )
                score = float(COMPONENTS["brute_force_model"].predict(dmatrix)[0])
                card = builder.build(
                    "brute_force", f"203.0.113.{200 + i}", "192.168.1.5",
                    score, COMPONENTS["brute_force_threshold"], shap, "brute_force",
                )
                ALERT_CACHE[card.alert_id] = card.to_dict()

            elif i % 3 == 2 and "dga_explainer" in COMPONENTS:
                dga_sample = np.random.RandomState(i).uniform(0, 1, (1, 9))
                dga_sample[0, 0] = 3.8
                dga_sample[0, 1] = 4.5
                dga_sample[0, 5] = 50.0
                dga_sample[0, 6] = 0.9

                shap = COMPONENTS["dga_explainer"].explain(
                    dga_sample, "dga",
                    context={"source_ip": f"10.0.5.{50 + i}"}
                )
                dmatrix = xgb.DMatrix(
                    dga_sample,
                    feature_names=COMPONENTS["dga_explainer"].feature_names,
                )
                score = float(COMPONENTS["dga_model"].predict(dmatrix)[0])
                card = builder.build(
                    "dga", f"10.0.5.{50 + i}", None,
                    score, COMPONENTS["dga_threshold"], shap, "dga",
                )
                ALERT_CACHE[card.alert_id] = card.to_dict()

    # Р¤РёР»СЊС‚СЂСѓРµРј: СѓР±РёСЂР°РµРј РёРіРЅРѕСЂРёСЂРѕРІР°РЅРЅС‹Рµ Рё Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅРЅС‹Рµ
    result = []
    for alert in ALERT_CACHE.values():
        if alert["alert_id"] not in IGNORED_ALERTS:
            # РџРѕРјРµС‡Р°РµРј Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅРЅС‹Рµ
            if alert["source_ip"] in BLOCKED_IPS:
                alert["blocked"] = True
            result.append(alert)

    return result


def refresh_cache():
    """РџСЂРёРЅСѓРґРёС‚РµР»СЊРЅРѕ РѕР±РЅРѕРІР»СЏРµС‚ РєРµС€."""
    global ALERT_CACHE
    ALERT_CACHE = {}
    return generate_demo_alerts(use_cache=False)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# API Endpoints
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Р“Р»Р°РІРЅР°СЏ СЃС‚СЂР°РЅРёС†Р° РґР°С€Р±РѕСЂРґР°."""
    return DASHBOARD_HTML


@app.get("/api/alerts")
async def api_alerts(count: int = Query(default=10, le=50)):
    """API: РїРѕР»СѓС‡РёС‚СЊ СЃРїРёСЃРѕРє Р°Р»РµСЂС‚РѕРІ."""
    alerts = generate_demo_alerts(count)
    return JSONResponse({
        "alerts": alerts,
        "count": len(alerts),
        "blocked_ips": list(BLOCKED_IPS),
        "ignored_count": len(IGNORED_ALERTS),
        "timestamp": time.time(),
    })


@app.get("/api/health")
async def health():
    """API: РїСЂРѕРІРµСЂРєР° Р·РґРѕСЂРѕРІСЊСЏ."""
    return {
        "status": "ok",
        "detectors": {
            "beaconing": "beaconing_model" in COMPONENTS,
            "brute_force": "brute_force_model" in COMPONENTS,
            "dga": "dga_model" in COMPONENTS,
        },
        "blocked_ips": len(BLOCKED_IPS),
        "ignored_alerts": len(IGNORED_ALERTS),
        "timestamp": time.time(),
    }


@app.get("/api/alert/{alert_id}")
async def api_alert_detail(alert_id: str):
    """API: РґРµС‚Р°Р»Рё РѕРґРЅРѕРіРѕ Р°Р»РµСЂС‚Р° (РёР· РєРµС€Р°)."""
    if alert_id in ALERT_CACHE:
        alert = ALERT_CACHE[alert_id]
        # Р”РѕР±Р°РІР»СЏРµРј СЃС‚Р°С‚СѓСЃ
        alert["blocked"] = alert["source_ip"] in BLOCKED_IPS
        alert["ignored"] = alert_id in IGNORED_ALERTS
        return JSONResponse(alert)
    return JSONResponse({"error": "alert not found"}, status_code=404)


@app.post("/api/block/{alert_id}")
async def api_block_ip(alert_id: str):
    """
    API: Р·Р°Р±Р»РѕРєРёСЂРѕРІР°С‚СЊ IP РёР· Р°Р»РµСЂС‚Р°.
    Р’РѕР·РІСЂР°С‰Р°РµС‚ РєРѕРјР°РЅРґСѓ РґР»СЏ РєРѕРїРёСЂРѕРІР°РЅРёСЏ (human-in-the-loop).
    """
    global BLOCKED_IPS

    if alert_id not in ALERT_CACHE:
        return JSONResponse({"error": "alert not found"}, status_code=404)

    alert = ALERT_CACHE[alert_id]
    ip = alert["source_ip"]

    # РџСЂРѕРІРµСЂСЏРµРј, РЅРµ Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ Р»Рё СѓР¶Рµ
    if ip in BLOCKED_IPS:
        return JSONResponse({
            "success": False,
            "message": f"IP {ip} СѓР¶Рµ Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ",
            "ip": ip,
        })

    # Р‘Р»РѕРєРёСЂСѓРµРј (РІ РїР°РјСЏС‚Рё)
    BLOCKED_IPS.add(ip)

    # Р¤РѕСЂРјРёСЂСѓРµРј РєРѕРјР°РЅРґСѓ РґР»СЏ С„Р°РµСЂРІРѕР»Р°
    reason = alert.get("alert_type", "unknown")
    explanations = alert.get("explanations", [])
    if explanations:
        reason += " вЂ” " + explanations[0].get("explanation_short", "")

    fw_cmd = FirewallCommand(ip=ip, reason=reason, duration_hours=24)

    logger.info(f"IP Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ: {ip} (alert={alert_id})")

    return JSONResponse({
        "success": True,
        "message": f"IP {ip} Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ",
        "ip": ip,
        "alert_id": alert_id,
        "command": fw_cmd.iptables,
        "command_windows": fw_cmd.windows_firewall,
        "note": "IP Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ РІ РїР°РјСЏС‚Рё СЃРµСЂРІРµСЂР°. Р”Р»СЏ СЂРµР°Р»СЊРЅРѕР№ Р±Р»РѕРєРёСЂРѕРІРєРё РІС‹РїРѕР»РЅРёС‚Рµ РєРѕРјР°РЅРґСѓ РЅР° С„Р°РµСЂРІРѕР»Рµ.",
    })


@app.post("/api/ignore/{alert_id}")
async def api_ignore_alert(alert_id: str):
    """API: РёРіРЅРѕСЂРёСЂРѕРІР°С‚СЊ Р°Р»РµСЂС‚."""
    global IGNORED_ALERTS

    if alert_id not in ALERT_CACHE:
        return JSONResponse({"error": "alert not found"}, status_code=404)

    IGNORED_ALERTS.add(alert_id)
    logger.info(f"РђР»РµСЂС‚ РїСЂРѕРёРіРЅРѕСЂРёСЂРѕРІР°РЅ: {alert_id}")

    return JSONResponse({
        "success": True,
        "message": "РђР»РµСЂС‚ РѕС‚РјРµС‡РµРЅ РєР°Рє РёРіРЅРѕСЂРёСЂРѕРІР°РЅРЅС‹Р№",
        "alert_id": alert_id,
    })


@app.post("/api/refresh")
async def api_refresh():
    """API: РїСЂРёРЅСѓРґРёС‚РµР»СЊРЅРѕ РѕР±РЅРѕРІРёС‚СЊ Р°Р»РµСЂС‚С‹."""
    alerts = refresh_cache()
    return JSONResponse({
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": time.time(),
    })


@app.post("/api/reset")
async def api_reset():
    """API: СЃР±СЂРѕСЃРёС‚СЊ РІСЃРµ Р±Р»РѕРєРёСЂРѕРІРєРё Рё РёРіРЅРѕСЂРёСЂРѕРІР°РЅРёСЏ."""
    global BLOCKED_IPS, IGNORED_ALERTS, ALERT_CACHE
    BLOCKED_IPS = set()
    IGNORED_ALERTS = set()
    ALERT_CACHE = {}
    return JSONResponse({
        "success": True,
        "message": "Р’СЃРµ Р±Р»РѕРєРёСЂРѕРІРєРё Рё РёРіРЅРѕСЂРёСЂРѕРІР°РЅРёСЏ СЃР±СЂРѕС€РµРЅС‹",
    })


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# HTML Р”Р°С€Р±РѕСЂРґ
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clarify вЂ” Autonomous Security Layer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .header {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 16px 24px;
            display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; gap: 8px;
        }
        .header h1 { font-size: 20px; color: #58a6ff; }
        .header .status { font-size: 12px; color: #8b949e; }
        .header .status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
        .header .status .dot.green { background: #3fb950; }
        .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        .stat-card {
            background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px;
        }
        .stat-card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-card .value { font-size: 28px; font-weight: 600; margin-top: 4px; }
        .stat-card .value.critical { color: #f85149; }
        .stat-card .value.high { color: #d2991d; }
        .stat-card .value.normal { color: #3fb950; }
        .stat-card .value.info { color: #58a6ff; }
        .alert-list { display: flex; flex-direction: column; gap: 12px; }
        .alert-card {
            background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px;
            transition: border-color 0.2s, opacity 0.3s;
        }
        .alert-card:hover { border-color: #58a6ff; }
        .alert-card.critical { border-left: 3px solid #f85149; }
        .alert-card.high { border-left: 3px solid #d2991d; }
        .alert-card.medium { border-left: 3px solid #58a6ff; }
        .alert-card.blocked { opacity: 0.5; border-left: 3px solid #3fb950; }
        .alert-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; flex-wrap: wrap; gap: 4px; }
        .alert-type { font-weight: 600; font-size: 14px; text-transform: uppercase; }
        .alert-type.critical { color: #f85149; }
        .alert-type.high { color: #d2991d; }
        .alert-confidence { font-size: 12px; padding: 2px 8px; border-radius: 12px; background: #21262d; }
        .alert-source { font-size: 13px; color: #8b949e; margin-bottom: 8px; }
        .badge {
            display: inline-block; font-size: 10px; padding: 2px 6px; border-radius: 10px;
            margin-left: 6px; text-transform: uppercase; letter-spacing: 0.5px;
        }
        .badge.blocked { background: #3fb95033; color: #3fb950; border: 1px solid #3fb950; }
        .explanations { margin-top: 8px; }
        .explanation-item {
            display: flex; gap: 8px; padding: 4px 0; font-size: 13px;
            border-bottom: 1px solid #21262d;
        }
        .explanation-item:last-child { border-bottom: none; }
        .shap-value { font-family: monospace; font-size: 11px; min-width: 50px; }
        .shap-value.positive { color: #3fb950; }
        .shap-value.negative { color: #f85149; }
        .explanation-text { flex: 1; }
        .alert-actions { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
        .btn {
            padding: 6px 14px; border-radius: 6px; border: 1px solid #30363d;
            background: #21262d; color: #c9d1d9; cursor: pointer; font-size: 12px;
            transition: all 0.2s;
        }
        .btn:hover { background: #30363d; border-color: #58a6ff; }
        .btn.block { background: #da3633; border-color: #da3633; color: #fff; }
        .btn.block:hover { background: #f85149; }
        .btn.ignore { background: #21262d; border-color: #8b949e; color: #8b949e; }
        .btn.ignore:hover { background: #30363d; color: #c9d1d9; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .toast {
            position: fixed; bottom: 20px; right: 20px; padding: 12px 20px;
            border-radius: 8px; color: #fff; font-size: 14px; z-index: 1000;
            animation: slideIn 0.3s ease; max-width: 400px;
        }
        .toast.success { background: #238636; }
        .toast.error { background: #da3633; }
        .toast.info { background: #1f6feb; }
        @keyframes slideIn { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        .refresh-bar {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 16px; font-size: 12px; color: #8b949e; flex-wrap: wrap; gap: 8px;
        }
        .refresh-bar button {
            background: none; border: 1px solid #30363d; color: #8b949e;
            padding: 4px 12px; border-radius: 4px; cursor: pointer;
        }
        .refresh-bar button:hover { border-color: #58a6ff; color: #c9d1d9; }
        .command-block {
            background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
            padding: 12px; margin-top: 8px; font-family: monospace; font-size: 12px;
            color: #7ee787; display: none; overflow-x: auto;
        }
        .empty-state { text-align: center; padding: 48px; color: #8b949e; }
        @media (max-width: 768px) {
            .stats { grid-template-columns: repeat(2, 1fr); }
            .alert-header { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>рџ›Ў Clarify</h1>
        <div class="status">
            <span class="dot green"></span> <span id="detector-status">Р—Р°РіСЂСѓР·РєР°...</span>
            &nbsp;|&nbsp; рџљ« <span id="blocked-count">0</span> Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅРѕ
            &nbsp;|&nbsp; вњ… <span id="ignored-count">0</span> РїСЂРѕРёРіРЅРѕСЂРёСЂРѕРІР°РЅРѕ
        </div>
    </div>
    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <div class="label">РђР»РµСЂС‚РѕРІ</div>
                <div class="value critical" id="stat-total">вЂ”</div>
            </div>
            <div class="stat-card">
                <div class="label">РљСЂРёС‚РёС‡РµСЃРєРёС…</div>
                <div class="value critical" id="stat-critical">вЂ”</div>
            </div>
            <div class="stat-card">
                <div class="label">Р›Р°С‚РµРЅС‚РЅРѕСЃС‚СЊ</div>
                <div class="value normal" id="stat-latency">вЂ”</div>
            </div>
            <div class="stat-card">
                <div class="label">Р—Р°Р±Р»РѕРєРёСЂРѕРІР°РЅРѕ IP</div>
                <div class="value info" id="stat-blocked">вЂ”</div>
            </div>
        </div>
        <div class="refresh-bar">
            <span id="last-update">РџРѕСЃР»РµРґРЅРµРµ РѕР±РЅРѕРІР»РµРЅРёРµ: вЂ”</span>
            <div>
                <button onclick="loadAlerts()">рџ”„ РћР±РЅРѕРІРёС‚СЊ</button>
                <button onclick="resetAll()" style="color: #f85149;">рџ—‘ РЎР±СЂРѕСЃРёС‚СЊ РІСЃС‘</button>
            </div>
        </div>
        <div class="alert-list" id="alert-list">
            <div class="empty-state">Р—Р°РіСЂСѓР·РєР° Р°Р»РµСЂС‚РѕРІ...</div>
        </div>
    </div>

    <script>
        let currentAlerts = [];
        let blockedIPs = new Set();
        let ignoredAlerts = new Set();

        async function loadAlerts() {
            try {
                const resp = await fetch('/api/alerts?count=12');
                const data = await resp.json();
                currentAlerts = data.alerts || [];
                blockedIPs = new Set(data.blocked_ips || []);
                ignoredAlerts = new Set();
                renderAlerts(currentAlerts);
                updateStats(data);
                document.getElementById('blocked-count').textContent = blockedIPs.size;
                document.getElementById('ignored-count').textContent = data.ignored_count || 0;
                document.getElementById('last-update').textContent = 
                    'РћР±РЅРѕРІР»РµРЅРѕ: ' + new Date().toLocaleTimeString();
            } catch (e) {
                showToast('РћС€РёР±РєР° Р·Р°РіСЂСѓР·РєРё Р°Р»РµСЂС‚РѕРІ', 'error');
            }
        }

        function renderAlerts(alerts) {
            const container = document.getElementById('alert-list');
            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<div class="empty-state">вњ… РќРµС‚ Р°РєС‚РёРІРЅС‹С… Р°Р»РµСЂС‚РѕРІ</div>';
                return;
            }

            container.innerHTML = alerts.map(alert => {
                const severityClass = alert.severity || 'medium';
                const isBlocked = alert.blocked || blockedIPs.has(alert.source_ip);
                const cardClass = isBlocked ? severityClass + ' blocked' : severityClass;

                const explanations = (alert.explanations || []).map(exp => {
                    const shapClass = exp.shap_value > 0 ? 'positive' : 'negative';
                    const sign = exp.shap_value > 0 ? '+' : '';
                    return `
                        <div class="explanation-item">
                            <span class="shap-value ${shapClass}">${sign}${exp.shap_value.toFixed(2)}</span>
                            <span class="explanation-text">${exp.explanation_short || exp.explanation || 'вЂ”'}</span>
                        </div>`;
                }).join('');

                const time = alert.time_local || new Date(alert.timestamp * 1000).toLocaleString();
                const blockedBadge = isBlocked ? '<span class="badge blocked">Р—РђР‘Р›РћРљРР РћР’РђРќ</span>' : '';

                return `
                    <div class="alert-card ${cardClass}" id="card-${alert.alert_id}">
                        <div class="alert-header">
                            <span class="alert-type ${severityClass}">${alert.alert_type.toUpperCase()}${blockedBadge}</span>
                            <span class="alert-confidence">${alert.confidence || Math.round(alert.model_score * 100) + '%'}</span>
                        </div>
                        <div class="alert-source">
                            ${alert.source_ip} ${alert.target_ip ? 'в†’ ' + alert.target_ip : ''} В· ${time}
                        </div>
                        <div class="explanations">${explanations}</div>
                        <div class="alert-actions">
                            <button class="btn block" onclick="blockIP('${alert.alert_id}')" ${isBlocked ? 'disabled' : ''}>
                                ${isBlocked ? 'вњ“ Р—Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ' : 'рџљ« Р—Р°Р±Р»РѕРєРёСЂРѕРІР°С‚СЊ'}
                            </button>
                            <button class="btn ignore" onclick="ignoreAlert('${alert.alert_id}')">вњ… РРіРЅРѕСЂРёСЂРѕРІР°С‚СЊ</button>
                            <button class="btn" onclick="showDetails('${alert.alert_id}')">рџ“Љ SHAP-Р°РЅР°Р»РёР·</button>
                        </div>
                        <div class="command-block" id="cmd-${alert.alert_id}"></div>
                    </div>`;
            }).join('');
        }

        function updateStats(data) {
            const alerts = data.alerts || [];
            document.getElementById('stat-total').textContent = alerts.length;
            const critical = alerts.filter(a => a.severity === 'critical').length;
            document.getElementById('stat-critical').textContent = critical;
            const latencies = alerts.map(a => a.latency_ms || 0).filter(l => l > 0);
            const avgLat = latencies.length > 0 ? (latencies.reduce((a,b) => a+b, 0) / latencies.length).toFixed(1) : 'вЂ”';
            document.getElementById('stat-latency').textContent = avgLat + ' РјСЃ';
            document.getElementById('stat-blocked').textContent = data.blocked_ips ? data.blocked_ips.length : 0;
        }

        async function blockIP(alertId) {
            const alert = currentAlerts.find(a => a.alert_id === alertId);
            if (!alert) return;

            if (!confirm(`Р—Р°Р±Р»РѕРєРёСЂРѕРІР°С‚СЊ IP ${alert.source_ip}?\\n\\nРџСЂРёС‡РёРЅР°: ${alert.alert_type.toUpperCase()}\\nРћР±СЉСЏСЃРЅРµРЅРёРµ: ${(alert.explanations||[])[0]?.explanation_short || 'вЂ”'}`)) {
                return;
            }

            try {
                const resp = await fetch('/api/block/' + alertId, { method: 'POST' });
                const data = await resp.json();

                if (data.success) {
                    blockedIPs.add(data.ip);
                    showToast(`IP ${data.ip} Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ`, 'success');

                    // РџРѕРєР°Р·С‹РІР°РµРј РєРѕРјР°РЅРґСѓ
                    if (data.command) {
                        const cmdBlock = document.getElementById('cmd-' + alertId);
                        if (cmdBlock) {
                            cmdBlock.style.display = 'block';
                            cmdBlock.textContent = '$ ' + data.command;
                        }
                    }

                    // РћР±РЅРѕРІР»СЏРµРј РєР°СЂС‚РѕС‡РєСѓ
                    const card = document.getElementById('card-' + alertId);
                    if (card) card.classList.add('blocked');

                    updateStats({alerts: currentAlerts, blocked_ips: [...blockedIPs]});
                    document.getElementById('blocked-count').textContent = blockedIPs.size;
                } else {
                    showToast(data.message || 'РћС€РёР±РєР° Р±Р»РѕРєРёСЂРѕРІРєРё', 'error');
                }
            } catch (e) {
                showToast('РћС€РёР±РєР° РїСЂРё Р±Р»РѕРєРёСЂРѕРІРєРµ', 'error');
            }
        }

        async function ignoreAlert(alertId) {
            try {
                const resp = await fetch('/api/ignore/' + alertId, { method: 'POST' });
                const data = await resp.json();

                if (data.success) {
                    ignoredAlerts.add(alertId);
                    showToast('РђР»РµСЂС‚ РїСЂРѕРёРіРЅРѕСЂРёСЂРѕРІР°РЅ', 'info');

                    // РЈР±РёСЂР°РµРј РєР°СЂС‚РѕС‡РєСѓ РёР· СЃРїРёСЃРєР°
                    const card = document.getElementById('card-' + alertId);
                    if (card) {
                        card.style.opacity = '0';
                        setTimeout(() => {
                            card.remove();
                            // РћР±РЅРѕРІР»СЏРµРј СЃС‚Р°С‚РёСЃС‚РёРєСѓ
                            const remaining = document.querySelectorAll('.alert-card').length;
                            document.getElementById('stat-total').textContent = remaining;
                        }, 300);
                    }

                    document.getElementById('ignored-count').textContent = ignoredAlerts.size;
                }
            } catch (e) {
                showToast('РћС€РёР±РєР° РїСЂРё РёРіРЅРѕСЂРёСЂРѕРІР°РЅРёРё', 'error');
            }
        }

        function showDetails(alertId) {
            window.open('/api/alert/' + alertId, '_blank');
        }

        async function resetAll() {
            if (!confirm('РЎР±СЂРѕСЃРёС‚СЊ РІСЃРµ Р±Р»РѕРєРёСЂРѕРІРєРё Рё РёРіРЅРѕСЂРёСЂРѕРІР°РЅРёСЏ?')) return;

            try {
                await fetch('/api/reset', { method: 'POST' });
                blockedIPs.clear();
                ignoredAlerts.clear();
                document.getElementById('blocked-count').textContent = '0';
                document.getElementById('ignored-count').textContent = '0';
                await loadAlerts();
                showToast('Р’СЃС‘ СЃР±СЂРѕС€РµРЅРѕ', 'info');
            } catch (e) {
                showToast('РћС€РёР±РєР° СЃР±СЂРѕСЃР°', 'error');
            }
        }

        function showToast(message, type) {
            const toast = document.createElement('div');
            toast.className = 'toast ' + type;
            toast.textContent = message;
            document.body.appendChild(toast);
            setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
        }

        async function loadHealth() {
            try {
                const resp = await fetch('/api/health');
                const data = await resp.json();
                const detectors = data.detectors || {};
                const active = Object.values(detectors).filter(v => v).length;
                document.getElementById('detector-status').textContent = active + ' РґРµС‚РµРєС‚РѕСЂРѕРІ';
                document.getElementById('blocked-count').textContent = data.blocked_ips || 0;
                document.getElementById('ignored-count').textContent = data.ignored_alerts || 0;
            } catch(e) {}
        }

        // Р—Р°РіСЂСѓР·РєР° РїСЂРё СЃС‚Р°СЂС‚Рµ
        loadAlerts();
        loadHealth();
        // РђРІС‚РѕРѕР±РЅРѕРІР»РµРЅРёРµ РєР°Р¶РґС‹Рµ 30 СЃРµРєСѓРЅРґ
        setInterval(loadAlerts, 30000);
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Clarify Web UI Server")
    parser.add_argument("--port", type=int, default=8000, help="РџРѕСЂС‚")
    parser.add_argument("--host", default="127.0.0.1", help="РҐРѕСЃС‚")
    parser.add_argument("--lang", default="ru", choices=["ru", "en"], help="РЇР·С‹Рє")
    args = parser.parse_args()

    global COMPONENTS
    COMPONENTS = init_components(args.lang)

    logger.info(f"Clarify Web UI Р·Р°РїСѓС‰РµРЅ: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

