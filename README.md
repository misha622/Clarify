
# Clarify Autonomous Security Layer

**Explainable ML layer for autonomous response — on top of your existing SIEM.**

Clarify connects to your Wazuh/ELK/Suricata and adds:
- **SHAP explanations** for every alert — see *why* the model flagged an event
- **Human-in-the-loop response** — confirm-flow with webhook or CLI command, never autonomous by default
- **Open-core** — detectors and explainers are free; premium connectors and support available

> ⚠️ **Status: MVP / Pre-alpha.** Not for production. Seeking 3–5 pilot users for real-world testing.

---

## Quick Start (5 minutes)

```bash
# 1. Clone
git clone https://github.com/misha622/Clarify.git
cd clarify

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 3. Train the Beaconing detector (synthetic data)
python -m src.models.train_beaconing

# 4. Run the demo
python -m src.ui.alert_card
```

You'll see an alert card with SHAP explanations in your terminal.

---

## Example Usage

### 1. Train a detector

```bash
# Russian explanations
python -m src.models.train_beaconing --lang ru

# English explanations
python -m src.models.train_beaconing --lang en

# With custom parameters
python -m src.models.train_beaconing --hosts 500 --cv 10
```

**Output:**
```
INFO: Train: 2847 samples, attack=312 (11.0%)
INFO: Cross-validation (5 folds)...
INFO:   Fold 1: F1=0.987, Precision=0.993, Recall=0.981, Threshold=0.856
INFO:   Fold 2: F1=0.992, Precision=0.995, Recall=0.989, Threshold=0.912
INFO:   ...
INFO:   Mean: F1=0.989±0.003, Precision=0.994, Recall=0.984
INFO: Final model trained in 0.3s
INFO: Threshold calibrated: 0.904 (precision=1.000, recall=1.000)
INFO: Model saved: models/beaconing_xgb.json
```

### 2. Run detection + explanation

```python
from src.detectors.beaconing import BeaconingDetector
from src.explainers.shap_explainer import ShapExplainer
from src.rendering.template_renderer import TemplateRenderer
from src.ui.alert_card import AlertCardBuilder, AlertCardRenderer

# Load model
import xgboost as xgb
model = xgb.Booster()
model.load_model("models/beaconing_xgb.json")

# Initialize components
explainer = ShapExplainer(model, feature_names=[...], top_n=3)
renderer = TemplateRenderer("config/feature_dictionary.yaml")
builder = AlertCardBuilder(template_renderer=renderer)
cli = AlertCardRenderer()

# Detect
detector = BeaconingDetector()
result = detector.detect(timestamps, source_id="45.33.32.156")

if result.is_alert:
    # Explain
    shap_result = explainer.explain(feature_vector, "beaconing", context={"source_ip": "45.33.32.156"})
    
    # Build alert card
    card = builder.build(
        alert_type="beaconing",
        source_ip="45.33.32.156",
        target_ip="10.0.5.17",
        model_score=result.score,
        model_threshold=detector.decision_threshold,
        shap_explanation=shap_result,
    )
    
    # Render
    print(cli.render(card))
    
    # Or get JSON for API
    print(card.to_json())
```

**Output (CLI):**
```
╔════════════════════════════════════════════════════════════╗
║ ⛔ BEACONING
║ Source: 45.33.32.156 → 10.0.5.17
║ Time: 2026-06-19 06:19:06
║ Confidence: 99%
╠════════════════════════════════════════════════════════════╣
║ WHY IT TRIGGERED (model explanation):
║ ▲ [3.18] Low interval variability (σ=6.6s) — suspicious for beaconing
║ ▲ [1.25] Intervals suspiciously regular (CV=0.05) — consistent with C2
║ ▲ [0.24] Interval distribution abnormally ordered (H=1.8) — automated C2
╠════════════════════════════════════════════════════════════╣
║ THREAT INTEL:
║   (threat intel sources not connected)
╠════════════════════════════════════════════════════════════╣
║ ACTIONS:
║  [BLO] Block IP 45.33.32.156
║  [IGN] Ignore
║  [SHO] Detailed SHAP analysis
╚════════════════════════════════════════════════════════════╝
  ⏱ Analysis completed in 1.1 ms
```

### 3. Block an IP (confirm-flow)

```python
from src.ui.confirm_flow import ConfirmFlow, WebhookConfig

# Option A: With webhook
webhook = WebhookConfig(url="https://firewall.internal/api/block")
flow = ConfirmFlow(webhook_config=webhook)

# Validate webhook first
valid, msg = flow.validate_webhook()
print(f"Webhook: {msg}")

# Block
result = flow.execute_block(
    ip="203.0.113.45",
    reason="Brute-force RDP (Clarify SHAP: CV=0.05, σ=6.6s)",
    alert_id="alert-beaconing-001",
)

# Option B: Without webhook — copy command
flow_no_webhook = ConfirmFlow()
result = flow_no_webhook.execute_block(
    ip="203.0.113.45",
    reason="C2 beaconing detected",
    alert_id="alert-beaconing-002",
)
print(f"Command to run:\n$ {result.command}")
# Output: $ iptables -A INPUT -s 203.0.113.45 -j DROP -m comment --comment "Clarify: C2 beaconing detected"
```

### 4. Docker

```bash
# Build
docker build -t clarify .

# Run with model mounted
docker run --rm -v $(pwd)/models:/app/models clarify

# Or with docker-compose
docker-compose up
```

---

## Architecture

```
Your logs (Wazuh/ELK/Suricata)
         │
         ▼
   ┌─────────────┐
   │  Connectors  │  ← read events (Wazuh connector ready, ELK planned)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Detectors   │  ← XGBoost on engineered features
   │  (Beaconing, │
   │   BruteForce,│
   │   DGA, UEBA) │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ TreeExplainer│  ← exact SHAP values (< 2 ms)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ NL Templates │  ← "Intervals suspiciously regular (CV=0.05)"
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Alert Card   │  ← CLI / JSON / Web UI
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Confirm-flow │  ← webhook or copy command, not autonomous
   └─────────────┘
```

---

## Components

| Component | Description | Status |
|-----------|-------------|--------|
| `detectors/beaconing.py` | C2 beaconing detector (window stats) | ✅ Ready |
| `detectors/dga.py` | DNS DGA (lexical features) | ✅ Ready |
| `detectors/brute_force.py` | RDP/SSH brute-force | ✅ Ready |
| `explainers/shap_explainer.py` | TreeExplainer for XGBoost | ✅ Ready |
| `rendering/template_renderer.py` | NL templates from YAML dictionary | ✅ Ready |
| `ui/alert_card.py` | CLI + JSON alert card | ✅ Ready |
| `ui/confirm_flow.py` | Human-in-the-loop blocking | ✅ Ready |
| `api/server.py` | FastAPI dashboard | ✅ Ready |
| `connectors/wazuh_connector.py` | Log ingestion from Wazuh | ✅ Ready |
| `data/synthetic_generator.py` | Synthetic data for training | ✅ Ready |
| `models/train_beaconing.py` | Training + CV + calibration | ✅ Ready |
| UEBA detector | Behavioral anomalies | 📋 Planned |

---

## Configuration

Detection thresholds: `config/detectors.yaml`
```yaml
detectors:
  beaconing:
    min_intervals: 15
    decision_threshold: 0.904  # auto-calibrated
    window_size_seconds: 3600
```

NL templates (RU): `config/feature_dictionary.yaml`  
NL templates (EN): `config/feature_dictionary_en.yaml`

```yaml
f_beacon_001:
  human_name: "Coefficient of Variation of Intervals"
  nl_templates:
    - condition: "value is not None"
      template: "Intervals suspiciously regular (CV={value:.2f}) — consistent with C2 beaconing"
      template_short: "Regular intervals (CV={value:.2f})"
```

---

## For Pilot Users

The model is trained on synthetic data and calibrated for demonstration. Real-world accuracy will be lower. **During the first 2 weeks of the pilot, I manually review every alert with you.**

Contact: [your contact info]

---

## License

Core detectors and explainers: MIT.  
Enterprise connectors: proprietary (on request).

---

## Roadmap

- [x] Beaconing detector + SHAP + NL templates
- [x] CLI alert card + JSON API
- [x] Confirm-flow (webhook / copy command)
- [x] English localization
- [x] Cross-validation training
- [x] Docker
- [x] DGA detector
- [x] Brute-force detector
- [x] Wazuh connector
- [x] Web UI (FastAPI + HTMX)
- [ ] Threat intel (AbuseIPDB)
- [ ] Multi-tenancy
```


git commit -m "Fix: README fully updated — component statuses, roadmap, test count, architecture"
git push
```
