#!/bin/bash
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
cd /Users/lucaaliberti/Downloads/xsage-clean
LOG=outputs_results/logs/overnight_battery_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
say(){ echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
say "########## OVERNIGHT BATTERIA B_full (ml1m + mind, 5 seed) ##########"
say "=== ml1m 5 seed ==="; $XPY scripts/ml1m/battery_bfull.py ml1m 5; say "ml1m rc=$?"
say "=== mind 5 seed ===";  $XPY scripts/ml1m/battery_bfull.py mind 5; say "mind rc=$?"
say "########## FINE — CSV: battery_bfull_ml1m.csv + battery_bfull_mind.csv ; log: $LOG ##########"
