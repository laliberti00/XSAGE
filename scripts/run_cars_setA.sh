#!/usr/bin/env bash
# Set A baseline CITABILI montate nella battery uniforme, sui 3 dataset winner.
# Backbone aggiunti: EASE [Steck 2019], SASRec [Kang 2018], FPMC [Rendle 2010] — ognuno
# valutato DA SOLO e +X-SAGE (SIT/Steck-a/Steck-b/UNI) con lo stesso identico protocollo di ml-1m.
#
# Vincoli (NON violare): generatori (venv-cornac, mps) UNO alla volta; MAI un generatore mentre
# gira la battery. ml1m: BFULL_SAFE=1 + matrici extra in memory-map → RAM-safe.
# NB: ogni step scrive su un log dedicato; il riassunto si estrae DAL FILE (niente pipe live → grep).
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

  echo "[$CITY] (1/4) gen EASE ..."
  $CPY scripts/cars/gen_scores_ease.py "$CITY" > "logs/setA_${CITY}_ease.log" 2>&1
  grep -e "->" -e "EASE lamb" "logs/setA_${CITY}_ease.log" | tail -2

  echo "[$CITY] (2/4) gen SASRec (ep=$SAS_EP) ..."
  $CPY scripts/cars/gen_scores_sasrec.py "$CITY" mps "$SAS_EP" > "logs/setA_${CITY}_sasrec.log" 2>&1
  grep -e "->" -e "scorate" "logs/setA_${CITY}_sasrec.log" | tail -1

  echo "[$CITY] (3/4) gen FPMC (ep=$SAS_EP) ..."
  $CPY scripts/cars/gen_scores_fpmc.py "$CITY" mps "$SAS_EP" > "logs/setA_${CITY}_fpmc.log" 2>&1
  grep -e "->" -e "scorate" "logs/setA_${CITY}_fpmc.log" | tail -1

  echo "[$CITY] (4/4) battery + EASE,SASRec,FPMC montati ($SEEDS seed) ..."
  SAFE=""; [ "$CITY" = "ml1m" ] && SAFE="BFULL_SAFE=1"
  env $SAFE XTRA_BACKBONES=EASE,SASRec,FPMC $XPY scripts/ml1m/battery_bfull.py "$CITY" "$SEEDS" > "logs/setA_${CITY}_battery.log" 2>&1
  grep -e "HEADLINE" -e "dCatMRR" -e "dR@20" -e "TOST" -e "backbone extra" "logs/setA_${CITY}_battery.log"
  echo "[$CITY] battery log completo: logs/setA_${CITY}_battery.log"
done
echo ""; echo "===== SET A COMPLETATO — outputs_results/battery_bfull_<city>.csv (B_blind,B_full,EASE,SASRec,FPMC) ====="
