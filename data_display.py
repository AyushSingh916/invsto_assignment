import psycopg2

# PostgreSQL connection details
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "mydb"
DB_USER = "myuser"
DB_PASSWORD = "mypassword"

def fetch_data():
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

        # Query the stock data from the database
        cursor.execute("SELECT * FROM stock_data")
        rows = cursor.fetchall()

        # Display the fetched data
        print("Fetched data from the database:")
        for row in rows:
            print(row)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    fetch_data()
