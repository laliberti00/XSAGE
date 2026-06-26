#!/bin/bash
# ============================================================================
# PROTOCOLLO UNIFORME k-core=10 (identico a ml-1m) — ricostruisce DA ZERO i dataset
# che erano a k-core=20 (yelp, kuairand, amazon), per un confronto VALIDO.
# Per ognuno: preprocess(k-core10) -> backbone BPR -> close_params -> battery 5 seed
#             -> macro_avg -> neutrality_ablation. Alla fine: aligned_matrix.
#
# ml-1m e mind sono GIA' k-core10 -> non toccati (riusati nella matrice).
# Foursquare (nyc/tokyo) -> port separato dal RAW TSMC2014 (taxonomy fine->root): TODO a parte.
#
# Lancia:  bash scripts/run_kcore10_all.sh
# Log:     logs/k10_<city>_<stage>.log
#
# ⚠️ MEMORIA: la battery B_full scorre l'intero catalogo. Amazon (catalogo grande) ha gia'
#    saturato la RAM una volta. Sono in ordine sicuro->rischioso (amazon ultimo). Se freeza
#    su amazon, gli altri sono gia' salvati; rilancia solo amazon a Mac fresco.
# ============================================================================
set -u
CLEAN=/Users/lucaaliberti/Downloads/xsage-clean
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
CPY=/Users/lucaaliberti/Downloads/.venv-cornac/bin/python
cd "$CLEAN"; mkdir -p logs

run_dataset(){
  CITY=$1; shift; PREP="$@"
  L=logs/k10_$CITY
  echo ""; echo "############################ $CITY (k-core10) ############################"
  echo "[$CITY] (1/6) preprocess k-core10..."
  $XPY $PREP > ${L}_prep.log 2>&1 || { echo "  !! PREP FAIL ($CITY) — salto"; return; }
  tail -3 ${L}_prep.log | sed 's/^/      /'
  echo "[$CITY] (2/6) backbone BPR..."
  mkdir -p data/$CITY/backbone
  $CPY scripts/mind/cornac_backbone.py $CITY > ${L}_bb.log 2>&1 || { echo "  !! BACKBONE FAIL — salto"; return; }
  echo "[$CITY] (3/6) close_params (ri-chiusura sui nuovi dati)..."
  rm -f outputs_results/params/$CITY.json          # forza ri-chiusura: gli indici sono cambiati
  $XPY scripts/ml1m/close_params.py $CITY 2>&1 | tee ${L}_close.log >/dev/null
  echo "      params: $(cat outputs_results/params/$CITY.json 2>/dev/null | tr -d '\n ')"
  echo "[$CITY] (4/6) battery_bfull 5 seed..."
  $XPY scripts/ml1m/battery_bfull.py $CITY 5 2>&1 | tee ${L}_batt.log | grep -E "HEADLINE|dCatMRR|dR@20|TOST" | sed 's/^/      /'
  echo "[$CITY] (5/6) macro_avg..."
  $XPY scripts/yelp/macro_avg.py $CITY 2>&1 | tee ${L}_macro.log | grep -E "saturazione|κ\*|Δ\(SIT" | sed 's/^/      /'
  echo "[$CITY] (6/6) neutrality_ablation..."
  $XPY scripts/yelp/neutrality_ablation.py $CITY 2>&1 | tee ${L}_neut.log | grep -E "VALORE" | sed 's/^/      /'
}

START=$(date +%s)
# ordine sicuro -> rischioso (catalogo crescente; amazon ultimo)
run_dataset yelp     scripts/yelp/preprocess_yelp.py Philadelphia 10
run_dataset kuairand scripts/kuairand/preprocess_kuairand.py 10
run_dataset amazoncd scripts/amazon/preprocess_amazon.py 10

echo ""; echo "############################ MATRICE FINALE (k-core10 uniforme) ############################"
$XPY scripts/foursquare/aligned_matrix.py 2>&1 | tee logs/k10_matrix.log
echo ""; echo "[FINE] tempo totale: $(( ($(date +%s)-START)/60 )) min"
