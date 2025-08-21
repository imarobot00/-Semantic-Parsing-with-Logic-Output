# Text-to-SQL Inference System - Usage Guide

## Quick Start

### 1. Installation
```bash
# Install required dependencies
pip install torch transformers datasets tqdm sentencepiece protobuf

# Navigate to project directory
cd "/path/to/AI Project"
```

### 2. Run the Demo
```bash
# Start the interactive demo
python demo_runner.py
```

### 3. Choose Your Mode

The demo offers 4 options:

1. **Run predefined examples** - See the system in action with curated examples
2. **Test custom question** - Enter your own questions interactively  
3. **Interactive demo** - Full featured interface with schema browsing
4. **Exit** - Quit the application

## Using the Inference System Programmatically

```python
from inference_system import TextToSQLInference

# Initialize the system
inference = TextToSQLInference()

# Get available databases
databases = inference.get_available_databases()
print("Available databases:", databases)

# Generate SQL for a question
result = inference.generate_sql(
    question="How many records are there?",
    db_id="perpetrator"  # Use any available database
)

if result['success']:
    print(f"Generated SQL: {result['generated_sql']}")
else:
    print(f"Error: {result['error']}")

# Get schema information
schema_info = inference.get_schema_info("perpetrator")
print("Tables:", schema_info['tables'])
```

## Understanding the Output

### Successful Generation
```python
{
    "question": "How many students are there?",
    "database_id": "university", 
    "generated_sql": "SELECT COUNT(*) FROM student;",
    "input_text": "question: How many students are there? schema: Tables: student(id, name, age)",
    "method": "picard",
    "success": True
}
```

### Error Case
```python
{
    "question": "Invalid question",
    "database_id": "unknown_db",
    "generated_sql": "",
    "error": "Database 'unknown_db' not found in schemas",
    "success": False
}
```

## Example Questions by Database Type

### University Database
- "How many students are there?"
- "What are the names of all students?"
- "Which courses have more than 3 credits?"
- "Show students older than 20"

### Company Database  
- "How many employees work here?"
- "What departments exist?"
- "Show all managers and their salaries"
- "Which employees joined this year?"

### E-commerce Database
- "How many products are available?"
- "What are the top selling items?"
- "Show orders from last month"
- "Which customers made more than 5 orders?"

## Tips for Better Results

1. **Be Specific**: Use clear, unambiguous language
2. **Match Schema**: Reference actual table/column names when possible
3. **Start Simple**: Begin with basic queries before trying complex ones
4. **Check Validation**: Always review the validation results
5. **Explore Schemas**: Use the schema browser to understand available data

## Troubleshooting

### Model Loading Issues
- The system automatically falls back to T5-small if trained model unavailable
- Ensure you have internet connection for downloading pre-trained models

### Schema Issues
- If Spider dataset unavailable, system uses mock university schema
- Check that `data/spider_data/tables.json` exists for full functionality

### Generation Problems
- Some questions may not map well to available schemas
- Try rephrasing questions to be more database-specific
- Use the interactive mode to see schema structure first

## Advanced Usage

### Custom Model Path
```python
inference = TextToSQLInference(
    model_path="path/to/your/trained/model",
    schemas_path="path/to/schemas.json",
    db_path="path/to/databases"
)
```

### Disable Picard Constraints
```python
result = inference.generate_sql(
    question="Your question",
    db_id="database_id", 
    use_picard=False  # Use standard beam search
)
```

### Adjust Generation Parameters
```python
result = inference.generate_sql(
    question="Your question",
    db_id="database_id",
    max_length=512,    # Longer SQL queries
    num_beams=10       # More beam search paths
)
```

## Learning Path

1. **Start with Demo**: Run `python demo_runner.py` to see examples
2. **Explore Schemas**: Use option 3 to browse available databases
3. **Try Simple Questions**: Start with count and selection queries
4. **Progress to Complex**: Move to joins and aggregations
5. **Study Output**: Examine generated SQL to understand patterns
6. **Experiment**: Try variations of successful questions

Happy querying!
