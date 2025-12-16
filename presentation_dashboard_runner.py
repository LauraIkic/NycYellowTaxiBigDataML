class TaxiOpsPresentationRunner:
    """
    Runner-Klasse (wie euer dashboard_runner.py), aber als Class.
    """

    def __init__(self, port: int = 8052, year_filter: int = 2025):
        self.port = port
        self.year_filter = year_filter

    def run(self) -> int:
        print("[PRESENTATION] Starting Taxi Ops Presentation Dashboard...")
        print(f"[PRESENTATION] URL: http://localhost:{self.port}")

        try:
            from presentation_dashboard import TaxiOpsPresentationDashboard

            dashboard = TaxiOpsPresentationDashboard(year_filter=self.year_filter)
            app = dashboard.create_app()
            app.run(debug=False, host="0.0.0.0", port=self.port)
            return 0

        except KeyboardInterrupt:
            print("\n[PRESENTATION] Stopped by user")
            return 0
        except Exception as e:
            print(f"[PRESENTATION][ERROR] Failed to start: {e}")
            return 1


if __name__ == "__main__":
    exit(TaxiOpsPresentationRunner(port=8052, year_filter=2025).run())