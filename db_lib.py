import sqlite3
from config import *


def create_table(connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS pickled_objects (
        id INTEGER PRIMARY KEY,
        data BLOB
    );
    """
    cursor = connection.cursor()
    cursor.execute(create_table_query)
    connection.commit()
pass

def migrate_data(source_conn, target_conn, table_name):
    # Retrieve data from the table in the source database
    source_cursor = source_conn.cursor()
    source_cursor.execute(f"SELECT * FROM {table_name}")
    data = source_cursor.fetchall()

    # Insert data into the corresponding table in the in-memory database
    target_cursor = target_conn.cursor()
    for row in data:
        target_cursor.execute(f"INSERT INTO {table_name} VALUES (?, ?)", row)

    target_conn.commit()
pass

def save_to_disk(source_conn, target_db_file):
    with open(target_db_file, "w"):
        pass  # Create an empty file or overwrite the existing file

    # Connect to the target SQLite database (on disk)
    target_conn = sqlite3.connect(target_db_file)

    # Create the table structure in the target database
    create_table(target_conn)

    # Retrieve data from the in-memory database
    source_cursor = source_conn.cursor()
    source_cursor.execute("SELECT * FROM pickled_objects")
    data = source_cursor.fetchall()

    # Insert data into the target database
    target_cursor = target_conn.cursor()
    for row in data:
        target_cursor.execute("INSERT INTO pickled_objects VALUES (?, ?)", row)

    target_conn.commit()
    target_conn.close()
pass

def init_memory_db():
    # Connect to the in-memory SQLite database
    conn_to_file = sqlite3.connect(db_in_file)
    conn_to_memory = sqlite3.connect(":memory:")

    # Create the table structure in the in-memory database
    create_table(conn_to_memory)
    create_table(conn_to_file)

    migrate_data(conn_to_file, conn_to_memory, table_name)
    conn_to_file.close()

    # Insert data into the in-memory database

    conn_to_memory.execute('PRAGMA journal_mode = OFF;')
    cursor_to_memory = conn_to_memory.cursor()

    return conn_to_memory, cursor_to_memory
pass

import mysql.connector


def create_training_table(db_config=db_config, table_name=table_name):
    
    if SQL_TYPE == 1:
        # Connect to the SQLite database or create a new one if it doesn't exist
        conn = sqlite3.connect(db_in_file)

        # Create a table to store the pickled objects
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS pickled_objects (id INTEGER PRIMARY KEY, data BLOB)')
        conn.commit()
        conn.close()
    elif SQL_TYPE == 2:
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            # Define the table structure
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                data LONGBLOB
            );
            """
            cursor.execute(create_table_query)
            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as e:
            print("Error:", e)
    pass
pass

def insert_training_data(training_data,db_config=db_config, table_name="pickled_objects"):
    if SQL_TYPE == 1:
      conn = sqlite3.connect(db_in_file)
      cursor = conn.cursor()
      cursor.executemany('INSERT INTO pickled_objects (data) VALUES (?)', [(data,) for data in training_data])
      conn.commit()
      conn.close()
    elif SQL_TYPE == 2:
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            
            # INSERT
            sql = 'INSERT INTO pickled_objects (data) VALUES (%s)'

            cursor.executemany(sql, [(data,) for data in training_data])
            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as e:
            print("Error:", e)
    pass

def create_files_table(db_config):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {MARIA_DB_MODULE_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_data LONGBLOB
        )
        """
        cursor.execute(create_table_query)

        print(f"Table '{MARIA_DB_MODULE_TABLE}' created successfully!")
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        print("Error:", e)


def clear_files_table(db_config):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        clear_table_query = f"TRUNCATE TABLE {MARIA_DB_MODULE_TABLE}"
        cursor.execute(clear_table_query)

        print(f"Table '{MARIA_DB_MODULE_TABLE}' cleared successfully!")

    except mysql.connector.Error as e:
        print("Error:", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def store_file_in_mariadb(file_path, db_config):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        with open(file_path, "rb") as file:
            file_data = file.read()

        # Check if the 'files' table already has data
        cursor.execute(f"SELECT COUNT(*) FROM {MARIA_DB_MODULE_TABLE}")
        num_rows = cursor.fetchone()[0]

        if num_rows > 0:
            # If the table already has data, update the latest file_data
            query = f"UPDATE {MARIA_DB_MODULE_TABLE} f1 INNER JOIN (SELECT MAX(id) AS max_id FROM {MARIA_DB_MODULE_TABLE} FOR UPDATE) f2 " \
                      "ON f1.id = f2.max_id SET f1.file_data = %s"
            cursor.execute(query, (file_data,))
            print("File updated successfully in MariaDB!")
        else:
            # If the table is empty, insert a new record
            query = f"INSERT INTO {MARIA_DB_MODULE_TABLE} (file_data) VALUES (%s)"
            cursor.execute(query, (file_data,))
            print("File stored successfully in MariaDB!")

        conn.commit()

    except mysql.connector.Error as e:
        print("Error:", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def load_file_from_mariadb(file_path, db_config):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        query = f"SELECT file_data FROM {MARIA_DB_MODULE_TABLE} ORDER BY id DESC LIMIT 1"
        cursor.execute(query)

        file_data = cursor.fetchone()[0]

        with open(file_path, "wb") as file:
            file.write(file_data)

        print("File loaded from MariaDB!")

    except mysql.connector.Error as e:
        print("Error:", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":

    try:
        conn_to_memory, cursor_to_memory = init_memory_db()

        # Execute the SELECT COUNT(*) query to get the number of records
        cursor_to_memory.execute('SELECT COUNT(*) FROM pickled_objects')

        # Fetch the result (since we're using COUNT(*), there will be only one row with the count)
        record_count = cursor_to_memory.fetchone()[0]

        print(record_count)

        # Save the in-memory database to disk
        save_to_disk(conn_to_memory, db_out_file)

        print(f"In-memory SQLite database saved to {db_in_file}.")
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        # Close the connection
        if conn_to_memory:
            conn_to_memory.close()
        pass
    pass
pass