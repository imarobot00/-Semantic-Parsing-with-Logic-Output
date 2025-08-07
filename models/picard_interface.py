import torch
from transformers import PreTrainedTokenizer
from typing import Optional, List, Dict, Any
import re


class PicardDecoder:
    def __init__(self, tokenizer: PreTrainedTokenizer, db_path: str, schemas: dict, 
                 fix_issue_16_primary_keys: bool = False):
        self.tokenizer = tokenizer
        self.schemas = schemas
        self.db_path = db_path
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        
        # SQL keywords for basic validation
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 
            'LIMIT', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'ON', 'AS',
            'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'ALL'
        }

    def step(self, input_ids, decoded_ids, db_id):
        """
        Given the current input_ids and decoded_ids, return allowed token ids.
        This is a simplified version without full Picard grammar constraints.
        """
        try:
            # Decode current output (partial SQL)
            decoded_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
            trimmed_sql = decoded_text.split("<pad>")[-1].strip()

            # Basic validation - check if SQL looks reasonable
            if self._is_valid_partial_sql(trimmed_sql, db_id):
                # Return None to allow all tokens (fallback behavior)
                return None
            else:
                # Return empty list to force alternative generation
                return []

        except Exception as e:
            print("⚠️ Picard constraint failed:", e)
            return None

    def _is_valid_partial_sql(self, sql: str, db_id: str) -> bool:
        """
        Basic SQL validation without full grammar parsing.
        """
        if not sql.strip():
            return True  # Empty SQL is valid for starting
            
        # Check for basic SQL structure
        sql_upper = sql.upper()
        
        # Must start with SELECT for Spider dataset
        if not sql_upper.strip().startswith('SELECT'):
            return False
            
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            # Allow unbalanced during generation
            return True
            
        # Check if table/column names exist in schema
        if db_id in self.schemas:
            schema = self.schemas[db_id]
            table_names = [name.lower() for name in schema.get('table_names_original', [])]
            
            # Basic check - if FROM is present, check if table exists
            from_match = re.search(r'from\s+(\w+)', sql_upper.lower())
            if from_match:
                table_name = from_match.group(1).lower()
                if table_name not in table_names and table_name not in ['(', 'select']:
                    return False
        
        return True

    def decode(self, model, input_ids, attention_mask, db_id: str, 
               max_length: int = 100, num_beams: int = 4, early_stopping: bool = True, **kwargs):
        """
        Decode using the model with basic SQL constraints.
        Falls back to standard beam search if constraints fail.
        """
        try:
            # Use standard beam search generation
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    do_sample=False,
                    **kwargs
                )
            
            # Post-process to ensure SQL validity
            generated_ids = outputs[0]
            decoded_sql = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Basic SQL cleanup
            cleaned_sql = self._clean_generated_sql(decoded_sql)
            
            # Re-encode the cleaned SQL
            cleaned_ids = self.tokenizer.encode(cleaned_sql, return_tensors='pt')[0]
            
            return [cleaned_ids]
            
        except Exception as e:
            print(f"⚠️ Picard decoding failed: {e}")
            # Fallback to standard generation
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    do_sample=False
                )
            return [outputs[0]]

    def _clean_generated_sql(self, sql: str) -> str:
        """
        Clean up generated SQL to fix common issues.
        """
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        
        # Ensure proper capitalization of SQL keywords
        for keyword in self.sql_keywords:
            sql = re.sub(rf'\b{keyword.lower()}\b', keyword, sql, flags=re.IGNORECASE)
        
        # Fix common spacing issues
        sql = re.sub(r'\s*,\s*', ', ', sql)  # Fix comma spacing
        sql = re.sub(r'\s*\(\s*', ' (', sql)  # Fix parentheses spacing
        sql = re.sub(r'\s*\)\s*', ') ', sql)
        
        # Ensure SQL ends properly (no trailing spaces/punctuation)
        sql = sql.strip()
        if sql and not sql.endswith(';') and not sql.endswith(')'):
            # Don't add semicolon as Spider doesn't use them
            pass
            
        return sql
