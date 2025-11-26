#!/usr/bin/env python3
"""
Orchestrator für das NYC Taxi Projekt
Verwaltet Startup, Restart und Dashboard-Start
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

def install_requirements():
    """Python Dependencies installieren"""
    print("[SETUP] Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[SETUP] Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install dependencies")
        sys.exit(1)

def start_docker():
    """Docker Services starten"""
    print("[SETUP] Starting Docker services...")
    try:
        subprocess.check_call(["docker", "compose", "up", "-d"])
        print("[SETUP] Docker services started!")
        time.sleep(5)  # Warten bis DB bereit ist
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to start Docker services")
        sys.exit(1)

def stop_services():
    """Alle Services stoppen"""
    print("[STOP] Stopping all services...")

    # Docker stoppen
    subprocess.run(["docker", "compose", "down", "-v"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Dashboard Prozesse beenden
    subprocess.run(["pkill", "-f", "python.*dashboard"],
                   stderr=subprocess.DEVNULL)
    subprocess.run(["lsof", "-ti:8051", "|", "xargs", "kill", "-9"],
                   shell=True, stderr=subprocess.DEVNULL)

def run_etl():
    """ETL Pipeline ausführen"""
    print("[ETL] Starting ETL pipeline...")
    try:
        subprocess.check_call([sys.executable, "main.py"])
        print("[ETL] Pipeline completed successfully!")
    except subprocess.CalledProcessError:
        print("[ERROR] ETL pipeline failed")
        sys.exit(1)

def start_dashboard():
    """Dashboard starten"""
    print("[DASHBOARD] Starting dashboard...")
    try:
        subprocess.check_call([sys.executable, "dashboard_runner.py"])
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to start dashboard")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Project Runner")
    parser.add_argument("--setup", action="store_true", help="Run initial setup")
    parser.add_argument("--restart", action="store_true", help="Restart all services")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--etl-only", action="store_true", help="Run only ETL without dashboard")

    args = parser.parse_args()

    if args.stop:
        stop_services()
        return

    if args.restart:
        stop_services()
        time.sleep(2)

    if args.setup or args.restart:
        install_requirements()
        start_docker()

    # ETL ausführen
    run_etl()

    # Dashboard starten (außer bei --etl-only)
    if not args.etl_only:
        start_dashboard()

if __name__ == "__main__":
    main()