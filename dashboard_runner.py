#!/usr/bin/env python3
"""
Dashboard Runner für NYC Taxi Projekt
Startet das Dash Dashboard auf Port 8051
"""

from dashboard import create_app

def main():
    """Startet das NYC Taxi Dashboard"""
    print("[DASHBOARD] Starting NYC Taxi Dashboard...")
    print("[DASHBOARD] Dashboard will be available at http://localhost:8051")
    print("[DASHBOARD] Press Ctrl+C to stop the dashboard")

    try:
        app = create_app()
        app.run(debug=False, host='0.0.0.0', port=8051)
    except KeyboardInterrupt:
        print("\n[DASHBOARD] Dashboard stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start dashboard: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
