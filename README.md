# Predictive Maintenance Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://symbotic-predictive-maintenance-ml-project.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

> **Live Deployed Application:** [https://symbotic-predictive-maintenance-ml-project.streamlit.app/](https://symbotic-predictive-maintenance-ml-project.streamlit.app/)

---

## Project Overview

The **Predictive Maintenance Intelligence Platform** is an end-to-end industrial IoT and machine learning system designed to monitor equipment telemetry, forecast machine failure risks before breakdown, and diagnose root causes in real-time.

By transitioning from reactive or calendar-based servicing to condition-based, data-driven predictive maintenance, the platform minimizes unplanned operational downtime, extends tooling lifecycle, and provides actionable dispatch recommendations for plant floor engineers.

---

## Key Capabilities

- **Real-Time Telemetry & Batch Ingestion:** Ingests live sensor streams and batch machine readings (rotational speed, torque, tool wear, process/air temperatures).
- **Physics-Informed Feature Engineering:** Derives real-time physical degradation indicators, including mechanical power output, thermal differentials ($\Delta T$), and load-wear stress interactions.
- **Cost-Optimized ML Failure Scoring:** Utilizes an optimized XGBoost classifier trained with positive class weighting to handle heavy class imbalance (3.4% baseline failure rate), evaluated against Precision-Recall AUC (PR-AUC) rather than misleading raw accuracy.
- **Root-Cause Failure Mode Diagnostics:** Pinpoints specific industrial failure mechanisms, including:
  - **HDF** (Heat Dissipation Failure)
  - **PWF** (Power Failure)
  - **OSF** (Overstrain Failure)
  - **TWF** (Tool Wear Failure)
  - **RNF** (Random / Transient Failures)
- **Interactive Decision Cockpit:** Provides an intuitive web interface for fleet telemetry tracking, risk-ranked maintenance queues, threshold calibration, and automated mitigation planning.

---

## Live Demo

Experience the interactive dashboard and test live failure predictions:

👉 **[Launch Predictive Maintenance Platform](https://symbotic-predictive-maintenance-ml-project.streamlit.app/)**
