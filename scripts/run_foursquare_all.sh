#!/bin/bash
# ============================================================================
# Foursquare nel protocollo UNIFORME (stesso flusso di ml-1m): 5 città TIST2015 + 2 città TSMC2014.
# Per ognuna: preprocess -> backbone BPR -> close_params (FRESH) -> battery 5-seed -> macro_avg
#             -> neutrality -> explainability -> costi.
#  - TIST2015 (nyc_tist, tokyo_tist, saopaulo, bangkok, istanbul): da OLD-processed (già k-core10),
#    port via preprocess_foursquare.py (deriva prev_geohash5 + intent_last_cat_idx).
#  - TSMC2014 (tsmc_nyc, tsmc_tky): dal RAW, k-core10, root-macro via tassonomia, GEO.
# Alla fine: aligned_matrix. (Amazon NON è qui: vedi run_amazon_safe.sh.)
#
# Lancia:  bash scripts/run_foursquare_all.sh
# Log:     logs/fsq_<city>_<stage>.log
# ============================================================================
set -u
CLEAN=/Users/lucaaliberti/Downloads/xsage-clean
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
CPY=/Users/lucaaliberti/Downloads/.venv-cornac/bin/python
cd "$CLEAN"; mkdir -p logs outputs_results/explain

run_dataset(){
  CITY=$1; shift; PREP="$@"
  L=logs/fsq_$CITY
  echo ""; echo "############################ $CITY ############################"
  echo "[$CITY] (1/8) preprocess..."
  $XPY $PREP > ${L}_prep.log 2>&1 || { echo "  !! PREP FAIL — salto"; return; }
  tail -2 ${L}_prep.log | sed 's/^/      /'
  echo "[$CITY] (2/8) backbone BPR..."
  mkdir -p data/$CITY/backbone
  $CPY scripts/mind/cornac_backbone.py $CITY > ${L}_bb.log 2>&1 || { echo "  !! BACKBONE FAIL — salto"; return; }
  echo "[$CITY] (3/8) close_params (FRESH)..."
  rm -f outputs_results/params/$CITY.json
  $XPY scripts/ml1m/close_params.py $CITY 2>&1 | tee ${L}_close.log >/dev/null
  echo "      params: $(cat outputs_results/params/$CITY.json 2>/dev/null | tr -d '\n ')"
  echo "[$CITY] (4/8) battery_bfull 5 seed..."
  $XPY scripts/ml1m/battery_bfull.py $CITY 5 2>&1 | tee ${L}_batt.log | grep -E "dCatMRR|dR@20|TOST" | sed 's/^/      /'
  echo "[$CITY] (5/8) macro_avg..."
  $XPY scripts/yelp/macro_avg.py $CITY 2>&1 | tee ${L}_macro.log | grep -E "saturazione|κ\*|Δ\(SIT" | sed 's/^/      /'
  echo "[$CITY] (6/8) neutrality..."
  $XPY scripts/yelp/neutrality_ablation.py $CITY 2>&1 | tee ${L}_neut.log | grep -E "VALORE" | sed 's/^/      /'
  echo "[$CITY] (7/8) explainability..."
  $XPY scripts/yelp/situation_profiles.py    $CITY 2>/dev/null > outputs_results/explain/situation_profiles_$CITY.json && echo "      profiles ok"
  $XPY scripts/yelp/situation_space.py       $CITY 2>/dev/null > outputs_results/explain/situation_space_$CITY.json && echo "      space ok"
  $XPY scripts/yelp/situation_transitions.py $CITY 2>/dev/null > outputs_results/explain/situation_transitions_$CITY.json && echo "      L3 ok"
  echo "[$CITY] (8/8) costi..."
  $XPY scripts/yelp/profile_cost.py $CITY 2>/dev/null > outputs_results/explain/cost_$CITY.txt && echo "      cost ok"
}

START=$(date +%s)
# TIST2015 (da OLD-processed, già k-core10)
for C in nyc_tist tokyo_tist saopaulo bangkok istanbul; do
  run_dataset $C scripts/foursquare/preprocess_foursquare.py $C
done
# TSMC2014 (dal RAW, k-core10, root-macro)
run_dataset tsmc_nyc scripts/foursquare/preprocess_tsmc.py tsmc_nyc
run_dataset tsmc_tky scripts/foursquare/preprocess_tsmc.py tsmc_tky

echo ""; echo "############################ MATRICE FINALE ############################"
$XPY scripts/foursquare/aligned_matrix.py 2>&1 | tee logs/fsq_matrix.log
echo ""; echo "[FINE] tempo totale: $(( ($(date +%s)-START)/60 )) min"
