#!/usr/bin/env bash
# PATH A — SD multi-seed COERENTE: i 4 backbone allenabili (SASRec,FPMC,DeepFM,AFM) rigenerati
# PER-SEED (42–46), così ogni seed è una pipeline indipendente e la SD cattura tutta la varianza.
# EASE resta singolo (closed-form, deterministico → SD=0 per natura). B_full già si ri-allena per seed.
#
# DISK-SAFE: un seed alla volta → genera le matrici del seed, runna la battery di QUEL seed
# (append al CSV), poi CANCELLA le matrici. Picco ~5GB (ml1m) invece di ~26GB.
# Auto-detach: lanci una riga, si stacca.
#
# Uso:   bash scripts/run_cars_pathA.sh [city ...]     (default: nyc_tist saopaulo ml1m)
#        XTRA_STECKA=0 bash scripts/run_cars_pathA.sh ml1m     (più veloce: salta Steck-a sugli extra)
# Monitor: tail -f logs/cars_pathA.log     Stop: kill <PID>
SAS_EP="${SAS_EP:-20}"; DEEP_EP="${DEEP_EP:-12}"

if [ "${PATHA_DETACHED:-}" != "1" ]; then
  cd /Users/lucaaliberti/Downloads/xsage-clean || exit 1; mkdir -p logs
  PATHA_DETACHED=1 nohup bash "$0" "$@" > logs/cars_pathA.log 2>&1 &
  echo "▶ Path A (SD multi-seed) in background.  PID $!"
  echo "  Monitor: tail -f logs/cars_pathA.log   Stop: kill $!"
  exit 0
fi

set -u
cd /Users/lucaaliberti/Downloads/xsage-clean
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
CPY=/Users/lucaaliberti/Downloads/.venv-cornac/bin/python
CITIES="${*:-nyc_tist saopaulo ml1m}"
SEEDS="42 43 44 45 46"
XTRA="EASE,SASRec,FPMC,DeepFM,AFM"
echo "===== PATH A — città: $CITIES | seq ep=$SAS_EP deep ep=$DEEP_EP | XTRA_STECKA=${XTRA_STECKA:-1} | $(date '+%H:%M:%S') ====="
for CITY in $CITIES; do
  echo ""; echo "##################### $CITY — $(date '+%H:%M:%S') #####################"
  CSV="outputs_results/battery_bfull_${CITY}.csv"; B="data/${CITY}/backbone"
  [ -f "$CSV" ] && cp "$CSV" "${CSV}.preA.bak"      # backup risultati pre-Path A
  rm -f "$CSV"                                       # fresh: i 5 seed-run appenderanno
  echo "[$CITY] gen EASE (deterministico, una volta) ..."
  $CPY scripts/cars/gen_scores_ease.py "$CITY" > "logs/pathA_${CITY}_ease.log" 2>&1
  grep -e "->" "logs/pathA_${CITY}_ease.log" | tail -1
  SAFE=""; [ "$CITY" = "ml1m" ] && SAFE="BFULL_SAFE=1"
  for S in $SEEDS; do
    echo "[$CITY] --- SEED $S --- $(date '+%H:%M:%S')"
    echo "[$CITY][$S] gen SASRec/FPMC/DeepFM/AFM ..."
    $CPY scripts/cars/gen_scores_sasrec.py "$CITY" mps "$SAS_EP" "$S" > "logs/pathA_${CITY}_s${S}_gen.log" 2>&1
    $CPY scripts/cars/gen_scores_fpmc.py   "$CITY" mps "$SAS_EP" "$S" >> "logs/pathA_${CITY}_s${S}_gen.log" 2>&1
    $XPY scripts/cars/gen_scores_deep.py   "$CITY" DeepFM "$DEEP_EP" "$S" >> "logs/pathA_${CITY}_s${S}_gen.log" 2>&1
    $XPY scripts/cars/gen_scores_deep.py   "$CITY" AFM    "$DEEP_EP" "$S" >> "logs/pathA_${CITY}_s${S}_gen.log" 2>&1
    grep -e "->" "logs/pathA_${CITY}_s${S}_gen.log" | tail -4
    echo "[$CITY][$S] battery (1 seed, append) ..."
    env $SAFE ONLY_SEED="$S" APPEND_CSV=1 XTRA_BACKBONES="$XTRA" XTRA_STECKA="${XTRA_STECKA:-1}" \
        $XPY scripts/ml1m/battery_bfull.py "$CITY" 1 >> "logs/pathA_${CITY}_battery.log" 2>&1
    rm -f "$B"/SASRec.s${S}.* "$B"/FPMC.s${S}.* "$B"/DeepFM.s${S}.* "$B"/AFM.s${S}.*   # libera disco
    echo "[$CITY][$S] fatto, matrici seed $S rimosse."
  done
  echo "[$CITY] COMPLETO: $(awk -F, 'NR>1{print $1}' "$CSV" | sort -u | tr '\n' ' ') seed × $(awk -F, 'NR>1{print $2}' "$CSV" | sort -u | wc -l | tr -d ' ') backbone — $(date '+%H:%M:%S')"
done
echo ""; echo "===== PATH A COMPLETATO — $(date '+%H:%M:%S'). SD multi-seed coerente nei battery_bfull_<city>.csv ====="
