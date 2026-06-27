#!/usr/bin/env bash
# Baseline CITABILI montate nella battery uniforme, sui 3 dataset winner.
# Backbone aggiunti (ognuno DA SOLO e +X-SAGE, stesso protocollo di ml-1m):
#   EASE   [Steck 2019]   via Cornac   — CF statico per-utente (no contesto)
#   SASRec [Kang 2018]    via Cornac   — sequenziale SOTA (contesto = sequenza)
#   FPMC   [Rendle 2010]  via Cornac   — Markov personalizzato (contesto = item prec.)
#   DeepFM [Guo 2017]     in-harness   — CONTEXT-AWARE feature (stesse 10 feature di B_full) + DNN
#   AFM    [Xiao 2017]    in-harness   — CONTEXT-AWARE feature + attenzione   (reduction-to-FM verificato)
#
# Vincoli: generatori UNO alla volta; mai un generatore mentre gira la battery.
# Matrici score caricate in mmap dalla battery → RAM-safe anche su ml1m (più backbone per-richiesta).
# ml1m: BFULL_SAFE=1 automatico. Steck-a greedy sui nuovi backbone: XTRA_STECKA=1 (default, completo);
#       per un run ml1m molto più veloce metti XTRA_STECKA=0 (salta il greedy sui backbone extra).
#
# Uso:  bash scripts/run_cars_setA.sh [city ...]        (default: ml1m nyc_tist saopaulo)
#       SAS_EP=20 DEEP_EP=12 SEEDS=5 bash scripts/run_cars_setA.sh nyc_tist
set -u
cd /Users/lucaaliberti/Downloads/xsage-clean
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
CPY=/Users/lucaaliberti/Downloads/.venv-cornac/bin/python
SAS_EP="${SAS_EP:-20}"; DEEP_EP="${DEEP_EP:-12}"; SEEDS="${SEEDS:-5}"
XTRA="EASE,SASRec,FPMC,DeepFM,AFM"
CITIES="${*:-ml1m nyc_tist saopaulo}"
mkdir -p logs
echo "===== SET A — città: $CITIES | seq ep=$SAS_EP deep ep=$DEEP_EP | seeds=$SEEDS | XTRA_STECKA=${XTRA_STECKA:-1} ====="
for CITY in $CITIES; do
  echo ""; echo "########## $CITY ##########"

  echo "[$CITY] (1/6) gen EASE ..."
  $CPY scripts/cars/gen_scores_ease.py "$CITY" > "logs/setA_${CITY}_ease.log" 2>&1
  grep -e "->" -e "EASE lamb" "logs/setA_${CITY}_ease.log" | tail -2

  echo "[$CITY] (2/6) gen SASRec (ep=$SAS_EP) ..."
  $CPY scripts/cars/gen_scores_sasrec.py "$CITY" mps "$SAS_EP" > "logs/setA_${CITY}_sasrec.log" 2>&1
  grep -e "->" -e "scorate" "logs/setA_${CITY}_sasrec.log" | tail -1

  echo "[$CITY] (3/6) gen FPMC (ep=$SAS_EP) ..."
  $CPY scripts/cars/gen_scores_fpmc.py "$CITY" mps "$SAS_EP" > "logs/setA_${CITY}_fpmc.log" 2>&1
  grep -e "->" -e "scorate" "logs/setA_${CITY}_fpmc.log" | tail -1

  echo "[$CITY] (4/6) gen DeepFM (ep=$DEEP_EP) [context-aware] ..."
  $XPY scripts/cars/gen_scores_deep.py "$CITY" DeepFM "$DEEP_EP" > "logs/setA_${CITY}_deepfm.log" 2>&1
  grep -e "->" "logs/setA_${CITY}_deepfm.log" | tail -1

  echo "[$CITY] (5/6) gen AFM (ep=$DEEP_EP) [context-aware] ..."
  $XPY scripts/cars/gen_scores_deep.py "$CITY" AFM "$DEEP_EP" > "logs/setA_${CITY}_afm.log" 2>&1
  grep -e "->" "logs/setA_${CITY}_afm.log" | tail -1

  echo "[$CITY] (6/6) battery + $XTRA montati ($SEEDS seed) ..."
  SAFE=""; [ "$CITY" = "ml1m" ] && SAFE="BFULL_SAFE=1"
  env $SAFE XTRA_BACKBONES="$XTRA" XTRA_STECKA="${XTRA_STECKA:-1}" $XPY scripts/ml1m/battery_bfull.py "$CITY" "$SEEDS" > "logs/setA_${CITY}_battery.log" 2>&1
  grep -e "HEADLINE" -e "dCatMRR" -e "dR@20" -e "TOST" -e "backbone extra" "logs/setA_${CITY}_battery.log"
  echo "[$CITY] battery log completo: logs/setA_${CITY}_battery.log"
done
echo ""; echo "===== SET A COMPLETATO — battery_bfull_<city>.csv (B_blind,B_full,EASE,SASRec,FPMC,DeepFM,AFM) ====="
