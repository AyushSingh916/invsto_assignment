import psycopg2
import pandas as pd

# Load the CSV data into a DataFrame
data = pd.read_csv('./stock_data.csv')

# Convert the 'datetime' column to a proper datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# PostgreSQL connection details
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "mydb"
DB_USER = "myuser"
DB_PASSWORD = "mypassword"

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            datetime TIMESTAMP NOT NULL,
            close NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            open NUMERIC NOT NULL,
            volume INTEGER NOT NULL,
            instrument VARCHAR(50) NOT NULL
        )
    """)
    conn.commit()

    # Insert data into the table
    for _, row in data.iterrows():
        cursor.execute("""
            INSERT INTO stock_data (datetime, close, high, low, open, volume, instrument)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (row['datetime'], row['close'], row['high'], row['low'], row['open'], row['volume'], row['instrument']))
    
    conn.commit()
    print("Data successfully inserted into PostgreSQL!")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if conn:
        cursor.close()
        conn.close()
