-- Create the data warehouse database
CREATE DATABASE ny_taxi_dwh;

-- Create a user for the application
CREATE USER ny_taxi_user WITH PASSWORD 'taxi_password';
GRANT ALL PRIVILEGES ON DATABASE ny_taxi_dwh TO ny_taxi_user;

-- Connect to the new database and grant schema permissions
\c ny_taxi_dwh;
GRANT ALL ON SCHEMA public TO ny_taxi_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ny_taxi_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ny_taxi_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ny_taxi_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ny_taxi_user;