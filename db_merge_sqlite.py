import sqlite3

def copy_table_data(source_conn, target_conn, table_name):
    # Retrieve data from the source table
    source_cursor = source_conn.cursor()
    source_cursor.execute(f"SELECT * FROM {table_name}")
    data = source_cursor.fetchall()

    # Get the maximum 'id' value in the target table
    target_cursor = target_conn.cursor()
    target_cursor.execute(f"SELECT MAX(id) FROM {table_name}")
    max_id = target_cursor.fetchone()[0] or 0

    # Copy data and adjust the 'id' values
    for row in data:
        max_id += 1
        row = (max_id,) + row[1:]  # Assuming 'id' is the first column
        target_cursor.execute(f"INSERT INTO {table_name} VALUES ({','.join(['?'] * len(row))})", row)

    target_conn.commit()

# Example usage:
if __name__ == "__main__":
    source_db_path = "pickled_objects_2.db"
    target_db_path = "pickled_objects_3.db"
    table_name = "pickled_objects"

    try:
        # Connect to both databases
        source_conn = sqlite3.connect(source_db_path)
        target_conn = sqlite3.connect(target_db_path)

        # Attach the source database to the target connection
        target_conn.execute(f"ATTACH DATABASE '{source_db_path}' AS source_db")

        # Copy data from the source table to the corresponding table in the target database
        copy_table_data(source_conn, target_conn, table_name)

        print(f"Data from {table_name} in {source_db_path} has been copied to {table_name} in {target_db_path}.")
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        # Close connections
        if source_conn:
            source_conn.close()
        if target_conn:
            target_conn.close()

