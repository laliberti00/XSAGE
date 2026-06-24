#!/bin/bash
# ============================================================================
# ALLINEAMENTO TOTALE — stesso protocollo, dall'inizio alla fine, su OGNI dataset.
# Per ogni città:  [port + backbone BPR (solo Foursquare)]  →  close_params (Fase A)
#                  →  battery_bfull 5 seed (B_blind + B_full)  →  macro_avg  →  neutrality_ablation
# Alla fine: aligned_matrix.py  →  outputs_results/aligned_matrix.csv
#
# Lancia:   bash scripts/run_aligned_all.sh
# Log:      logs/aligned_<city>_<stage>.log   (tqdm visibile a schermo via tee)
#
# NOTE:
#  - MIND è GRANDE (101K utenti): close_params + battery sono LENTI (ore). È l'unico pesante.
#  - Per rifare solo alcune città, modifica CITIES qui sotto.
#  - SKIP_EXISTING=1 salta gli stage già prodotti (params/<city>.json, battery_bfull_<city>.csv, ecc.).
#  - B_full usa solo contesto temporale (no geo) su TUTTI i dataset → trattamento uniforme; SIT usa il
#    geo dove c'è (yelp + 5 Foursquare). Questo è voluto: B_full identico ovunque.
# ============================================================================
set -u
CLEAN=/Users/lucaaliberti/Downloads/xsage-clean
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
CPY=/Users/lucaaliberti/Downloads/.venv-cornac/bin/python
cd "$CLEAN"; mkdir -p logs

# 4 dataset CORE (la legge a 3 gate). mind per ultimo (lento ~1.5h).
# Per il tabellone COMPLETO con Foursquare, aggiungi: nyc_tist tokyo_tist saopaulo bangkok istanbul
CITIES="ml1m yelp kuairand mind"
FSQ="nyc_tist tokyo_tist saopaulo bangkok istanbul"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

is_fsq(){ for c in $FSQ; do [ "$c" = "$1" ] && return 0; done; return 1; }
have(){ [ "$SKIP_EXISTING" = "1" ] && [ -f "$1" ]; }

START=$(date +%s)
for CITY in $CITIES; do
  echo ""; echo "############################ $CITY ############################"
  L=logs/aligned_$CITY

  if is_fsq "$CITY"; then
    echo "[$CITY] (1/5) port Foursquare OLD→clean + backbone BPR..."
    $XPY scripts/foursquare/preprocess_foursquare.py "$CITY" > ${L}_port.log 2>&1 \
      || { echo "  !! PORT FAIL ($CITY) — salto"; continue; }
    $CPY scripts/mind/cornac_backbone.py "$CITY" > ${L}_backbone.log 2>&1 \
      || { echo "  !! BACKBONE FAIL ($CITY) — salto"; continue; }
  fi

  echo "[$CITY] (2/5) Fase A — close_params (chiusura parametri su val)..."
  if have "outputs_results/params/$CITY.json"; then echo "  (skip: params già presenti)";
  else $XPY scripts/ml1m/close_params.py "$CITY" 2>&1 | tee ${L}_close.log; fi

  echo "[$CITY] (3/5) battery_bfull 5 seed (B_blind + B_full, bootstrap/Holm/TOST)..."
  if have "outputs_results/battery_bfull_$CITY.csv"; then echo "  (skip: battery già presente)";
  else $XPY scripts/ml1m/battery_bfull.py "$CITY" 5 2>&1 | tee ${L}_battery.log; fi

  echo "[$CITY] (4/5) macro_avg (micro vs MACRO-averaged)..."
  if have "outputs_results/macro_avg_$CITY.csv"; then echo "  (skip: macro_avg già presente)";
  else $XPY scripts/yelp/macro_avg.py "$CITY" 2>&1 | tee ${L}_macro.log; fi

  echo "[$CITY] (5/5) neutrality_ablation..."
  if have "outputs_results/neutrality_ablation_$CITY.csv"; then echo "  (skip: ablazione già presente)";
  else $XPY scripts/yelp/neutrality_ablation.py "$CITY" 2>&1 | tee ${L}_neutrality.log; fi
done

echo ""; echo "############################ SITUAZIONI (explainability ml-1m) ############################"
mkdir -p outputs_results/explain
$XPY scripts/yelp/situation_profiles.py ml1m 2>/dev/null > outputs_results/explain/situation_profiles_ml1m.json && echo "  profiles ok"
$XPY scripts/yelp/situation_space.py ml1m 2>/dev/null > outputs_results/explain/situation_space_ml1m.json && echo "  space ok"
$XPY scripts/yelp/situation_transitions.py ml1m 2>/dev/null > outputs_results/explain/situation_transitions_ml1m.json && echo "  transitions(L3) ok"

echo ""; echo "############################ MATRICE FINALE ############################"
$XPY scripts/foursquare/aligned_matrix.py 2>&1 | tee logs/aligned_matrix.log
echo ""; echo "[FINE] tempo totale: $(( ($(date +%s)-START)/60 )) min"
