#!/usr/bin/env bash
# Set A baseline CITABILI montate nella battery uniforme, sui 3 dataset winner.
# Backbone aggiunti: EASE [Steck 2019], SASRec [Kang 2018], FPMC [Rendle 2010] — ognuno
# valutato DA SOLO e +X-SAGE (SIT/Steck-a/Steck-b/UNI) con lo stesso identico protocollo di ml-1m.
#
# Vincoli appresi (NON violare):
#  - generatori (venv-cornac, mps) UNO alla volta; MAI un generatore mentre gira la battery (crash mps).
#  - ml1m: BFULL_SAFE=1 (B_full in f16) + matrici extra in memory-map → RAM-safe.
#
# Uso:  bash scripts/run_cars_setA.sh [city ...]        (default: ml1m nyc_tist saopaulo)
#       SAS_EP=20 SEEDS=5 bash scripts/run_cars_setA.sh nyc_tist
set -u
cd /Users/lucaaliberti/Downloads/xsage-clean
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
CPY=/Users/lucaaliberti/Downloads/.venv-cornac/bin/python
SAS_EP="${SAS_EP:-20}"; SEEDS="${SEEDS:-5}"
CITIES="${*:-ml1m nyc_tist saopaulo}"
mkdir -p logs
echo "===== SET A — città: $CITIES | SASRec/FPMC ep=$SAS_EP | battery seeds=$SEEDS ====="
for CITY in $CITIES; do
  echo ""; echo "########## $CITY ##########"
  echo "[$CITY] (1/4) gen EASE ..."   ; $CPY scripts/cars/gen_scores_ease.py  "$CITY"            2>&1 | tee "logs/setA_${CITY}_ease.log"   | grep -E "->|EASE"
  echo "[$CITY] (2/4) gen SASRec ..." ; $CPY scripts/cars/gen_scores_sasrec.py "$CITY" mps $SAS_EP 2>&1 | tee "logs/setA_${CITY}_sasrec.log" | grep -E "->|scorate"
  echo "[$CITY] (3/4) gen FPMC ..."   ; $CPY scripts/cars/gen_scores_fpmc.py   "$CITY" mps $SAS_EP 2>&1 | tee "logs/setA_${CITY}_fpmc.log"   | grep -E "->|scorate"
  echo "[$CITY] (4/4) battery + EASE,SASRec,FPMC montati ($SEEDS seed) ..."
  SAFE=""; [ "$CITY" = "ml1m" ] && SAFE="BFULL_SAFE=1"
  env $SAFE XTRA_BACKBONES=EASE,SASRec,FPMC $XPY scripts/ml1m/battery_bfull.py "$CITY" "$SEEDS" 2>&1 | tee "logs/setA_${CITY}_battery.log" | grep -vE "it/s|%\|"
done
echo ""; echo "===== SET A COMPLETATO — outputs_results/battery_bfull_<city>.csv (B_blind,B_full,EASE,SASRec,FPMC) ====="
