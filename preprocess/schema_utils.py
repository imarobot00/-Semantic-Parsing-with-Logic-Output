import json
from typing import List, Dict

def load_tables(file_path: str) -> Dict[str, dict]:
    with open(file_path, 'r') as f:
        tables = json.load(f)
    return {table['db_id']: table for table in tables}

def flatten_schema(db_schema: dict) -> str:
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    
    # Init table → list of column names
    table_columns = {table_name: [] for table_name in table_names}
    
    for col_id, (table_id, col_name) in enumerate(column_names):
        if table_id == -1 or col_name == "*":
            continue  # Skip global columns like "*"
        table_name = table_names[table_id]
        table_columns[table_name].append(col_name)
    
    # Join into string
    flat_schema_parts = []
    for table, columns in table_columns.items():
        cols_formatted = ", ".join(columns)
        flat_schema_parts.append(f"{table}({cols_formatted})")
    
    flat_schema = "Tables: " + ", ".join(flat_schema_parts)
    return flat_schema


if __name__ == "__main__":
    # Update path if needed
    import os

    tables_path = os.path.join(os.path.dirname(__file__), "../data/spider_data/tables.json")
    tables_path = os.path.abspath(tables_path)

 
    all_tables = load_tables(tables_path)

    # Pick a database to test with
    db_id = "department_store"  
    schema = all_tables[db_id]
    print(flatten_schema(schema))
