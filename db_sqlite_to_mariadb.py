import sqlite3
import mysql.connector

source_db_path = "sampleDB/pickled_objects.db"
target_db_name = "pickled_objects.db"
table_name = "Log"

target_db_host = "tel123escope.ntunhs.edu.tw"
target_db_port = 1234  # Your MariaDB port
target_db_user = "asssg"
target_db_password = "sss"


def create_database(connection, db_name):
    # Create the database
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    connection.commit()

def create_table(connection, table_name):
    # Define the table structure
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        data LONGBLOB
    );
    """

    cursor = connection.cursor()
    cursor.execute(create_table_query)
    connection.commit()

def migrate_table_data(source_conn, target_conn, table_name):
    # Retrieve data from the SQLite table
    source_cursor = source_conn.cursor()
    source_cursor.execute(f"SELECT * FROM {table_name}")
    data = source_cursor.fetchall()

    # Insert data into the corresponding table in the MariaDB database
    target_cursor = target_conn.cursor()
    for row in data:
        target_cursor.execute(f"INSERT INTO {table_name} VALUES (NULL, %s)", (row[1],))

    target_conn.commit()

# Example usage:
if __name__ == "__main__":


    try:
        # Connect to the SQLite database
        source_conn = sqlite3.connect(source_db_path)

        # Connect to the MariaDB database
        target_conn = mysql.connector.connect(
            host=target_db_host,
            port=target_db_port,
            user=target_db_user,
            password=target_db_password,
        )

        # Create the target database
        create_database(target_conn, target_db_name)

        # Use the target database
        target_conn.database = target_db_name

        # Create the table structure in MariaDB
        create_table(target_conn, table_name)

        # Migrate data from the SQLite table to the corresponding table in the MariaDB database
        migrate_table_data(source_conn, target_conn, table_name)

        print(f"Data from {table_name} in SQLite database has been migrated to {table_name} in MariaDB database.")
    except (sqlite3.Error, mysql.connector.Error) as e:
        print(f"Error: {e}")
    finally:
        # Close connections
        if source_conn:
            source_conn.close()
        if target_conn:
            target_conn.close()
