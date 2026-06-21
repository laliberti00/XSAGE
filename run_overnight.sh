#!/usr/bin/env bash
# Lancia la selezione congiunta notturna. Lancia-e-dormi: checkpointed,
# resumable, idempotente. Rilanciarlo riprende da dove si era fermato.
#
#   ./run_overnight.sh
#
# Gira in foreground; per lasciarlo girare staccato dal terminale usa nohup
# (vedi sotto). Tutto l'output va sia a video sia in logs/overnight_<ts>.log.
set -euo pipefail

CLEAN="/Users/lucaaliberti/Downloads/xsage-clean"
PY="/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python"

cd "$CLEAN"
echo "Avvio selezione notturna — $(date)"
echo "Log in: logs/overnight_<timestamp>.log"
echo "Per il progresso: tail -f logs/overnight_*.log"
echo

# Esecuzione robusta: se cade, il rilancio riprende dai checkpoint su disco.
exec "$PY" -m scripts.overnight_selection
