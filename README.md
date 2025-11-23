# NYC Yellow Taxi Big Data ML Project

Data Engineering Projekt mit ETL Pipeline für NYC Yellow Taxi Daten. Lädt Parquet-Dateien in ein PostgreSQL Data Warehouse und stellt interaktive Dashboards bereit.

## Projektübersicht

- **ETL Pipeline**: Extrahiert, transformiert und lädt NYC Taxi-Daten
- **Data Warehouse**: PostgreSQL mit Stern-Schema (Fact/Dimension Tables)
- **Dashboard**: Interaktive Visualisierungen mit Dash/Plotly
- **Docker**: Containerisierte PostgreSQL-Datenbank

## Voraussetzungen

- Python 3.11+
- Docker & Docker Compose (muss gestartet sein)
- Git

## Installation & Erststart

```bash

# Komplett-Setup (Dependencies + Docker + ETL + Dashboard)
python run.py --setup
```

**Dashboard verfügbar unter:** http://localhost:8051

## Verwendung

### Normaler Lauf (wenn Docker bereits läuft)
```bash
python run.py
```

### Nur ETL Pipeline ausführen
```bash
python run.py --etl-only
# oder direkt:
python main.py
```

### Nur Dashboard starten (nach erfolgreichem ETL)
```bash
python dashboard_runner.py
```

### Alle Services neu starten
```bash
python run.py --restart
```

### Alle Services stoppen
```bash
python run.py --stop
```

## Projektstruktur

```
├── main.py                    # ETL Pipeline (Hauptprogramm)
├── run.py                     # Orchestrator Script
├── dashboard_runner.py        # Dashboard Starter
├── dashboard.py               # Dash App Definition
├── requirements.txt           # Python Dependencies
├── docker-compose.yml         # PostgreSQL Container
├── db_connection.py           # Datenbankverbindung
├── taxi_data_extractor.py     # Datenextraktion
├── taxi_data_transformer.py   # Datentransformation
├── taxi_data_cleaner.py       # Datenbereinigung
└── assets/                    # Parquet Dateien
    ├── yellow_tripdata_2025-01.parquet
    ├── yellow_tripdata_2025-02.parquet
    └── yellow_tripdata_2025-03.parquet
```

## Workflow (Minimale Schritte)

1. **Erstmaliger Start:**
   ```bash
   python run.py --setup
   ```

2. **Weitere Läufe:**
   ```bash
   python run.py
   ```

3. **Bei Datenstruktur-Änderungen:**
   ```bash
   python run.py --restart
   ```

## Services

- **PostgreSQL**: localhost:5432 (Container: 5433→5432)
- **Dashboard**: http://localhost:8051

## Troubleshooting

**Docker nicht gestartet:**
```bash
# Docker Desktop starten und warten
python run.py --setup
```

**Port bereits belegt:**
```bash
# Bestehende Services stoppen
python run.py --stop
# Dann neu starten
python run.py
```

**Datenbankverbindung fehlgeschlagen:**
```bash
# Container-Logs prüfen
docker compose logs postgres
# Services neu starten
python run.py --restart
```

## Beenden

```bash
python run.py --stop
```

Dies stoppt alle Docker Container und beendet das Dashboard.
