#!/bin/bash
# FASE A (chiusura parametri) -> FASE B (batteria due-assi, 5 seed). ml-1m. Difensivo.
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
cd /Users/lucaaliberti/Downloads/xsage-clean
LOG=outputs_results/logs/ml1m_overnight_$(date +%Y%m%d_%H%M%S).log
mkdir -p outputs_results/logs
exec > >(tee -a "$LOG") 2>&1
say(){ echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
say "########## ml-1m OVERNIGHT: FASE A -> FASE B ##########"
say "=== FASE A: chiusura parametri (val Cat-MRR + sensibilita) ==="
$XPY scripts/ml1m/close_params.py ml1m; rcA=$?; say "FASE A rc=$rcA"
if [ $rcA -ne 0 ]; then say "ATTENZIONE: FASE A fallita -> FASE B usera' i parametri DEFAULT (ereditati)"; fi
say "=== FASE B: batteria due-assi su B_full (5 seed) ==="
$XPY scripts/ml1m/battery_bfull.py ml1m 5; rcB=$?; say "FASE B rc=$rcB"
say "########## FINE — params/ml1m.json ; param_closure_ml1m.csv ; battery_bfull_ml1m.csv ; log=$LOG ##########"
