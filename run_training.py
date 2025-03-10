import os
import time
import subprocess

def run_training():
    print("Starting Automated Training...")
    subprocess.run(["python", "train.py"])

# Run training every 24 hours
while True:
    run_training()
    print("Next training in 24 hours...")
    time.sleep(86400)  # Wait 24 hours
