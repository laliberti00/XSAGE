#!/bin/bash
# ============================================================================
# AMAZON CDs&Vinyl — battery 5-seed MEMORY-SAFE (BFULL_SAFE=1) + macro + neutrality + explain + cost.
# Amazon ha già: processed (k-core10) + backbone + params CALIBRATI (close_params girato).
# La battery normale satura la RAM (matrice n_test×n_items ≈ 3.8GB) -> qui usa il provider LAZY
# (scoring a blocchi, RAM costante; più lento ma non freeza).
#
# Lancia (a Mac fresco, niente altro aperto):  bash scripts/run_amazon_safe.sh
# ============================================================================
set -u
CLEAN=/Users/lucaaliberti/Downloads/xsage-clean
XPY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
cd "$CLEAN"; mkdir -p logs outputs_results/explain
C=amazoncd; L=logs/amazon_safe

echo "### amazon: verifica artefatti pronti ###"
ls data/processed/$C/df_train.parquet data/$C/backbone/FM.scores.npy outputs_results/params/$C.json 2>&1 | sed 's/^/  /'

echo "### (1/4) battery 5 seed (MEMORY-SAFE) ###"
BFULL_SAFE=1 $XPY scripts/ml1m/battery_bfull.py $C 5 2>&1 | tee ${L}_batt.log | grep -E "SEED|dCatMRR|dR@20|TOST"
echo "### (2/4) macro_avg ###"
$XPY scripts/yelp/macro_avg.py $C 2>&1 | tee ${L}_macro.log | grep -E "saturazione|κ\*|Δ\(SIT"
echo "### (3/4) neutrality ###"
$XPY scripts/yelp/neutrality_ablation.py $C 2>&1 | tee ${L}_neut.log | grep -E "VALORE"
echo "### (4/4) explainability + cost ###"
$XPY scripts/yelp/situation_profiles.py    $C 2>/dev/null > outputs_results/explain/situation_profiles_$C.json && echo "  profiles ok"
$XPY scripts/yelp/situation_space.py       $C 2>/dev/null > outputs_results/explain/situation_space_$C.json && echo "  space ok"
$XPY scripts/yelp/situation_transitions.py $C 2>/dev/null > outputs_results/explain/situation_transitions_$C.json && echo "  L3 ok"
$XPY scripts/yelp/profile_cost.py          $C 2>/dev/null > outputs_results/explain/cost_$C.txt && echo "  cost ok"

echo "### matrice aggiornata ###"
$XPY scripts/foursquare/aligned_matrix.py 2>&1 | tail -8
