from db_cleaner import DBCleaner

# Connection details for your local database
cleaner = DBCleaner(
    user="Your_User_here",
    password="Your_PW_here",
    host="localhost",
    port="5432",
    dbname="ny_taxi_dwh",
)

# Names of tables you want to drop; adjust as needed
tables_to_drop = [
    "dim_date",
    "fact_trips",
]

cleaner.drop_tables(tables_to_drop)