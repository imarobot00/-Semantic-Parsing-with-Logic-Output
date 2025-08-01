import json
import re

def normalize_sql(sql):
    """Normalize SQL for comparison by removing extra spaces and standardizing formatting"""
    if not sql:
        return ''
    
    # Convert to lowercase and remove extra whitespace
    sql = ' '.join(sql.lower().split())
    
    # Standardize quote types (convert double quotes to single quotes)
    sql = sql.replace('"', "'")
    
    # Remove spaces around operators and punctuation
    sql = re.sub(r'\s*,\s*', ',', sql)
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    sql = re.sub(r'\s*=\s*', '=', sql)
    sql = re.sub(r'\s*<\s*', '<', sql)
    sql = re.sub(r'\s*>\s*', '>', sql)
    sql = re.sub(r'\s*<=\s*', '<=', sql)
    sql = re.sub(r'\s*>=\s*', '>=', sql)
    sql = re.sub(r'\s*!=\s*', '!=', sql)
    sql = re.sub(r'\s*<>\s*', '<>', sql)
    
    return sql.strip()

def main():
    print('🔍 Loading JSON file...')

    # Load the JSON file
    with open('output/predictions/t5_dev_predictions_spider.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f'📊 Total predictions: {len(data)}')

    # Check for exact matches
    exact_matches = 0
    updated_count = 0

    print('🔄 Scanning for exact matches...')

    for i, item in enumerate(data):
        if i % 500 == 0:
            print(f'  Processed {i}/{len(data)} entries...')
        
        query_norm = normalize_sql(item['query'])
        gold_norm = normalize_sql(item['gold'])
        
        if query_norm == gold_norm:
            if not item['exact_match']:
                item['exact_match'] = True
                updated_count += 1
            exact_matches += 1

    print(f'✅ Found {exact_matches} exact matches')
    print(f'🔄 Updated {updated_count} items to exact_match=True')

    # Save the updated JSON
    print('💾 Saving updated JSON file...')
    with open('output/predictions/t5_dev_predictions_spider.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print('✅ Updated JSON file saved successfully!')

    # Print some statistics
    accuracy = (exact_matches / len(data)) * 100
    print(f'📈 Exact match accuracy: {accuracy:.2f}% ({exact_matches}/{len(data)})')

    # Show a few examples of exact matches
    if exact_matches > 0:
        print('\n🔍 Sample exact matches found:')
        count = 0
        for item in data:
            if item['exact_match'] and count < 3:
                print(f'  Query: {item["query"]}')
                print(f'  Gold:  {item["gold"]}')
                print(f'  DB:    {item["db_id"]}')
                print()
                count += 1

if __name__ == "__main__":
    main()
