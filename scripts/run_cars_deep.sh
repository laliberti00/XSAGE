#!/usr/bin/env bash
# SOLO i nuovi backbone context-aware DeepFM+AFM, FUSI nei CSV esistenti (B_blind/B_full/EASE/
# SASRec/FPMC già fatti e committati restano intatti). NON rigenera nulla degli altri, NON ritocca
# B_full (SKIP_BFULL=1). Auto-detach: lanci una riga e si stacca.
#
# Flusso per città: gen DeepFM + gen AFM → battery SKIP_BFULL XTRA_BACKBONES=DeepFM,AFM (matrici in
# mmap, RAM-safe) → merge: righe DeepFM/AFM aggiunte al battery_bfull_<city>.csv esistente.
#
# Uso:   bash scripts/run_cars_deep.sh [city ...]      (default: nyc_tist saopaulo ml1m)
# Monitor: tail -f logs/cars_deep.log     Stop: kill <PID>
DEEP_EP="${DEEP_EP:-12}"; SEEDS="${SEEDS:-5}"

if [ "${DEEP_DETACHED:-}" != "1" ]; then
  cd /Users/lucaaliberti/Downloads/xsage-clean || exit 1; mkdir -p logs
  DEEP_DETACHED=1 nohup bash "$0" "$@" > logs/cars_deep.log 2>&1 &
  echo "▶ DeepFM+AFM (solo nuovi) in background.  PID $!"
  echo "  Monitor: tail -f logs/cars_deep.log   Stop: kill $!"
  exit 0
fi

set -u
cd /Users/lucaaliberti/Downloads/xsage-clean
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
CITIES="${*:-nyc_tist saopaulo ml1m}"
echo "===== DeepFM+AFM SOLO NUOVI — città: $CITIES | deep ep=$DEEP_EP seeds=$SEEDS | $(date '+%H:%M:%S') ====="
for CITY in $CITIES; do
  echo ""; echo "########## $CITY — $(date '+%H:%M:%S') ##########"
  CSV="outputs_results/battery_bfull_${CITY}.csv"

  echo "[$CITY] (1/4) gen DeepFM (ep=$DEEP_EP) ..."
  $XPY scripts/cars/gen_scores_deep.py "$CITY" DeepFM "$DEEP_EP" > "logs/deep_${CITY}_deepfm.log" 2>&1
  grep -e "->" "logs/deep_${CITY}_deepfm.log" | tail -1

  echo "[$CITY] (2/4) gen AFM (ep=$DEEP_EP) ..."
  $XPY scripts/cars/gen_scores_deep.py "$CITY" AFM "$DEEP_EP" > "logs/deep_${CITY}_afm.log" 2>&1
  grep -e "->" "logs/deep_${CITY}_afm.log" | tail -1

  echo "[$CITY] (3/4) battery SOLO DeepFM,AFM (no B_full retrain, $SEEDS seed) ..."
  [ -f "$CSV" ] && cp "$CSV" "${CSV}.bak"
  SAFE=""; [ "$CITY" = "ml1m" ] && SAFE="BFULL_SAFE=1"
  env $SAFE SKIP_BFULL=1 XTRA_BACKBONES=DeepFM,AFM XTRA_STECKA="${XTRA_STECKA:-1}" \
      $XPY scripts/ml1m/battery_bfull.py "$CITY" "$SEEDS" > "logs/deep_${CITY}_battery.log" 2>&1
  grep -e "HEADLINE: SIT-su-DeepFM" -e "HEADLINE: SIT-su-AFM" -e "dCatMRR" -e "dR@20" -e "TOST" "logs/deep_${CITY}_battery.log"

  echo "[$CITY] (4/4) merge righe DeepFM/AFM nel CSV esistente ..."
  $XPY - "$CITY" <<'PY'
import sys, pandas as pd
city = sys.argv[1]; base = f"outputs_results/battery_bfull_{city}.csv"
new = pd.read_csv(base)                                   # appena scritto: B_blind,DeepFM,AFM
add = new[new.backbone.isin(["DeepFM", "AFM"])]
try:
    old = pd.read_csv(base + ".bak")                      # esistente: B_blind,B_full,EASE,SASRec,FPMC
    old = old[~old.backbone.isin(["DeepFM", "AFM"])]      # evita doppioni su re-run
    final = pd.concat([old, add], ignore_index=True)
except FileNotFoundError:
    final = new
final.to_csv(base, index=False)
print(f"  [{city}] CSV finale: {sorted(final.backbone.unique())}  ({len(final)} righe)")
PY
done
echo ""; echo "===== DeepFM+AFM FUSI — $(date '+%H:%M:%S'). CSV: B_blind,B_full,EASE,SASRec,FPMC,DeepFM,AFM ====="
