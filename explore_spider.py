import json

# Open and load the JSON file
with open('data/spider_data/test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

count =0
# Explore each table schema
for table in data:
    print(f"\n=== Database: {table['db_id']} ===") # Database ID 
    count += 1
    print("Question:")
    print(f"{table['question']}")  # Question about the database
    print("Query:")
    print(f"{table['query']}")  # SQL query related to the database
    # for tname in table['question']:
    #     print(f"  Question : {tname}")

    # for tname in table['query']:
    #     print(f"  Query : {tname}")

    # print("\nColumns:")
    # for i, (table_idx, col_name) in enumerate(table['column_names']):
    #     if table_idx == -1:
    #         continue  # usually "*"
    #     table_name = table['table_names'][table_idx]
    #     col_type = table['column_types'][i]
    #     print(f"  [{table_name}] {col_name} ({col_type})")
        

    # print("\nPrimary Keys:", table['primary_keys'])
    # print("Foreign Keys:", table['foreign_keys'])

print(f"\nTotal databases explored: {count}")


#GPT generated code for exploring the Spider dataset
# The following code is a template for exploring the Spider dataset.
# Uncomment the following lines to use the code for exploring the Spider dataset.

# import json

# # --- Load JSON files ---
# def load_json(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# spider_path = "data/spider"
# train_data = load_json(f"{spider_path}/train_spider.json")
# tables_data = load_json(f"{spider_path}/tables.json")

# # --- Inspect one sample ---
# sample = train_data[0]  # you can change this index later

# db_id = sample['db_id']
# question = sample['question']
# sql = sample['query']

# print("=== Natural Language Question ===")
# print(question)
# print("\n=== SQL Query ===")
# print(sql)
# print("\n=== Database ID ===")
# print(db_id)

# # --- Get schema info from tables.json ---
# def get_schema_for_db(db_id, tables_data):
#     for db in tables_data:
#         if db['db_id'] == db_id:
#             return db
#     return None

# schema = get_schema_for_db(db_id, tables_data)
# if schema is None:
#     raise ValueError(f"No schema found for db_id {db_id}")

# # --- Print table names and columns ---
# print("\n=== Tables and Columns ===")
# for i, table_name in enumerate(schema['table_names']):
#     print(f"\nTable {i}: {table_name}")
#     columns = [col for col in schema['column_names'] if col[0] == i]
#     for col in columns:
#         _, col_name = col
#         print(f"  - {col_name}")
