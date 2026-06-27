#!/usr/bin/env bash
# Lancia in UNA riga tutta la Set A (EASE+SASRec+FPMC, gen + battery 5-seed) sui 3 dataset,
# in SEQUENZA (finisce una città, parte la dopo). Si auto-stacca in background: sopravvive
# alla chiusura del terminale. Ordine: piccole prima, ml1m (pesante) per ultima.
#
# Uso:   bash scripts/run_cars_all.sh
# Monitor: tail -f logs/cars_all.log
# Stop:    kill <PID stampato>     (oppure: pkill -f run_cars_all)

# --- auto-detach: se non già in background, si rilancia con nohup e esce ---
if [ "${CARS_DETACHED:-}" != "1" ]; then
  cd /Users/lucaaliberti/Downloads/xsage-clean || exit 1
  mkdir -p logs
  CARS_DETACHED=1 nohup bash "$0" "$@" > logs/cars_all.log 2>&1 &
  echo "▶ Set A lanciata in background.  PID $!"
  echo "  Monitor:  tail -f logs/cars_all.log"
  echo "  Stop:     kill $!"
  exit 0
fi

set -u
cd /Users/lucaaliberti/Downloads/xsage-clean
CITIES="${*:-nyc_tist saopaulo ml1m}"     # piccole prima, ml1m per ultima
echo "===== SET A SEQUENZIALE — città: $CITIES — avvio $(date '+%Y-%m-%d %H:%M:%S') ====="
for CITY in $CITIES; do
  echo ""
  echo "############################################################"
  echo "##  $CITY  —  START $(date '+%H:%M:%S')"
  echo "############################################################"
  bash scripts/run_cars_setA.sh "$CITY"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "##  $CITY  —  OK   $(date '+%H:%M:%S')"
  else
    echo "##  $CITY  —  FALLITA (rc=$rc) $(date '+%H:%M:%S') — continuo con la prossima"
  fi
done
echo ""
echo "===== TUTTO COMPLETATO — $(date '+%Y-%m-%d %H:%M:%S') ====="
echo "CSV: outputs_results/battery_bfull_{nyc_tist,saopaulo,ml1m}.csv  (5 backbone ciascuno)"
