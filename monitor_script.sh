#!/bin/zsh
PID=75609
RESTARTS=0
MAX_RESTARTS=5
JSON_PATH="/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments/neurips_revision.json"
LOG_PATH="/tmp/tg_rev2.log"
SCRIPT_PATH="/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/verify_neurips_revision.py"
START_TIME=$(date +%s)
MAX_TIME=$((25 * 60))

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [[ $ELAPSED -ge $MAX_TIME ]]; then
        echo "Exiting: Reached 25 minute wall time limit."
        break
    fi

    if [[ -f "$JSON_PATH" ]]; then
        echo "Exiting: $JSON_PATH exists."
        break
    fi

    ST=$(ps -p $PID -o pid,etime,stat 2>/dev/null || echo GONE)
    echo "--- PID $PID Status ---"
    echo "$ST"
    echo "--- Last 8 lines of log ---"
    tail -8 "$LOG_PATH"

    if [[ "$ST" == "GONE" ]]; then
        if [[ ! -f "$JSON_PATH" ]]; then
            if [[ $RESTARTS -lt $MAX_RESTARTS ]]; then
                echo "RESTARTING (Restart $((RESTARTS+1))/$MAX_RESTARTS)..."
                ( /opt/homebrew/bin/python3.11 -u "$SCRIPT_PATH" </dev/null >>"$LOG_PATH" 2>&1 & )
                sleep 5
                PID=$(pgrep -af verify_neurips_revision | grep "$SCRIPT_PATH" | head -1 | awk '{print $1}')
                RESTARTS=$((RESTARTS + 1))
                echo "New PID: $PID"
            else
                echo "Max restarts reached. Terminal state: GONE."
                break
            fi
        else
            echo "Process gone but JSON exists. Finishing."
            break
        fi
    fi

    echo "Sleeping 90 seconds (Total elapsed: $ELAPSED s)..."
    sleep 90
done

echo "=== FINAL REPORT ==="
ls -la /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments/
echo "--- Last 60 lines of log ---"
tail -60 "$LOG_PATH"

if [[ -f "$JSON_PATH" ]]; then
    echo "--- JSON Summary ---"
    /opt/homebrew/bin/python3.11 -c "import json; d=json.load(open('$JSON_PATH')); print(json.dumps(d['summary'], indent=2)); rb=d['real_bug']; print('REAL_BUG_TG=',rb['tensorguard']['verdict']); print('REAL_BUG_FX_OK=',rb['fx_shapeprop'].get('ok'),'err=',str(rb['fx_shapeprop'].get('err',''))[:200]); print('REAL_BUG_META_OK=',rb['meta_forward'].get('ok'),'err=',str(rb['meta_forward'].get('err',''))[:200])"
else
    echo "JSON not found."
fi
