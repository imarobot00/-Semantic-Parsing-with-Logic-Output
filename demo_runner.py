#!/usr/bin/env python3
"""
Simple Example Runner for Text-to-SQL Inference

This script demonstrates how to use the inference system with
predefined examples and shows the complete workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_system import TextToSQLInference


def run_examples():
    """Run predefined examples to demonstrate the system."""
    print("Running Text-to-SQL Examples...")
    
    # Initialize the inference system
    try:
        inference = TextToSQLInference()
    except Exception as e:
        print(f"Failed to initialize inference system: {e}")
        return
    
    # Get available databases
    databases = inference.get_available_databases()
    print(f"\nAvailable databases: {databases}")
    
    # Use the first available database (or mock 'university' database)
    db_id = databases[0] if databases else "university"
    
    # Show schema information
    print(f"\nSchema for '{db_id}':")
    schema_info = inference.get_schema_info(db_id)
    
    if "error" in schema_info:
        print(f"{schema_info['error']}")
        return
    
    for table, columns in schema_info['tables'].items():
        print(f"  {table}: {', '.join(columns)}")
    
    # Define example questions
    examples = [
        "How many students are there?",
        "What are the names of all students?",
        "Which courses have more than 3 credits?",
        "Show all students and their majors",
        "What is the average age of students?"
    ]
    
    print(f"\nTesting with example questions...")
    print("="*60)
    
    for i, question in enumerate(examples, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 40)
        
        # Generate SQL
        result = inference.generate_sql(question, db_id)
        
        if result['success']:
            print(f"Generated SQL: {result['generated_sql']}")
            
            # Validate SQL
            validation = inference.validate_sql(result['generated_sql'], db_id)
            if validation['valid']:
                print("Validation: PASSED")
            else:
                print(f"Validation: {validation['error']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
    print("Example run completed!")


def test_custom_question():
    """Test with a custom question."""
    print("\nTesting Custom Question...")
    
    try:
        inference = TextToSQLInference()
        
        # Get user input
        databases = inference.get_available_databases()
        print(f"Available databases: {databases}")
        
        db_id = input(f"Choose database ({'/'.join(databases)}): ").strip()
        if db_id not in databases:
            print(f"Using default database: {databases[0]}")
            db_id = databases[0]
        
        question = input("Enter your question: ").strip()
        
        if question:
            result = inference.generate_sql(question, db_id)
            
            print(f"\nInput: {result.get('input_text', '')}")
            print(f"Generated SQL: {result['generated_sql']}")
            
            if result['success']:
                validation = inference.validate_sql(result['generated_sql'], db_id)
                print(f"Valid: {validation['valid']}")
                if not validation['valid']:
                    print(f"Error: {validation['error']}")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function with menu options."""
    while True:
        print("\n" + "="*50)
        print("TEXT-TO-SQL INFERENCE DEMO")
        print("="*50)
        print("1. Run predefined examples")
        print("2. Test custom question")
        print("3. Interactive demo")
        print("4. Exit")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == "1":
            run_examples()
        elif choice == "2":
            test_custom_question()
        elif choice == "3":
            try:
                inference = TextToSQLInference()
                inference.interactive_demo()
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
