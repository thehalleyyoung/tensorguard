import subprocess
import time
import os
import json
import datetime

pid = 75609
restarts = 0
max_restarts = 5
json_path = "/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments/neurips_revision.json"
log_path = "/tmp/tg_rev2.log"
script_path = "/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/verify_neurips_revision.py"
python_path = "/opt/homebrew/bin/python3.11"
start_time = time.time()
max_time = 25 * 60

while True:
    elapsed = time.time() - start_time
    if elapsed >= max_time:
        print("Exiting: Reached 25 minute wall time limit.")
        break

    if os.path.exists(json_path):
        print(f"Exiting: {json_path} exists.")
        break

    try:
        # Check if process exists
        subprocess.check_output(["ps", "-p", str(pid)], stderr=subprocess.DEVNULL)
        st = subprocess.check_output(["ps", "-p", str(pid), "-o", "pid,etime,stat"], stderr=subprocess.DEVNULL).decode()
        print(f"--- PID {pid} Status ---")
        print(st)
    except subprocess.CalledProcessError:
        print(f"--- PID {pid} Status ---\nGONE")
        if not os.path.exists(json_path):
            if restarts < max_restarts:
                print(f"RESTARTING (Restart {restarts+1}/{max_restarts})...")
                with open(log_path, "a") as log_file:
                    proc = subprocess.Popen([python_path, "-u", script_path], stdout=log_file, stderr=subprocess.STDOUT)
                    pid = proc.pid
                restarts += 1
                print(f"New PID: {pid}")
            else:
                print("Max restarts reached. Terminal state: GONE.")
                break
        else:
            print("Process gone but JSON exists. Finishing.")
            break

    # Tail log
    try:
        log_tail = subprocess.check_output(["tail", "-8", log_path]).decode()
        print("--- Last 8 lines of log ---")
        print(log_tail)
    except:
        pass

    print(f"Sleeping 90 seconds (Total elapsed: {int(elapsed)} s)...")
    time.sleep(90)

print("=== FINAL REPORT ===")
subprocess.call(["ls", "-la", "/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments/"])
print("--- Last 60 lines of log ---")
subprocess.call(["tail", "-60", log_path])

if os.path.exists(json_path):
    print("--- JSON Summary ---")
    try:
        with open(json_path, 'r') as f:
            d = json.load(f)
            print(json.dumps(d['summary'], indent=2))
            rb = d['real_bug']
            print('REAL_BUG_TG=', rb['tensorguard']['verdict'])
            print('REAL_BUG_FX_OK=', rb['fx_shapeprop'].get('ok'), 'err=', str(rb['fx_shapeprop'].get('err',''))[:200])
            print('REAL_BUG_META_OK=', rb['meta_forward'].get('ok'), 'err=', str(rb['meta_forward'].get('err',''))[:200])
    except Exception as e:
        print(f"Error reading JSON: {e}")
else:
    print("JSON not found.")
