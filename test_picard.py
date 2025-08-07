#!/usr/bin/env python3
"""
Test script for PicardDecoder to verify it works correctly.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from models.picard_interface import PicardDecoder


def test_picard_decoder():
    """Test the PicardDecoder with a simple example."""
    print("🧪 Testing PicardDecoder...")
    
    # Initialize tokenizer and model
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    
    # Mock schemas
    mock_schemas = {
        "test_db": {
            "table_names_original": ["users", "orders"],
            "column_names_original": [
                [-1, "*"],
                [0, "id"],
                [0, "name"],
                [1, "id"],
                [1, "user_id"],
                [1, "total"]
            ]
        }
    }
    
    # Initialize PicardDecoder
    picard = PicardDecoder(
        tokenizer=tokenizer,
        schemas=mock_schemas,
        db_path="mock_path",
        fix_issue_16_primary_keys=True
    )
    
    # Test input
    input_text = "question: What are the names of all users? schema: Tables: users(id, name), orders(id, user_id, total)"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    
    print(f"📝 Input: {input_text}")
    
    # Test decode method
    try:
        outputs = picard.decode(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            db_id="test_db",
            max_length=50,
            num_beams=2,
            early_stopping=True
        )
        
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generated SQL: {generated_sql}")
        
        # Test step method
        decoded_ids = torch.tensor([1, 2, 3])  # Mock token ids
        valid_tokens = picard.step(input_ids, decoded_ids, "test_db")
        print(f"✅ Step method returned: {type(valid_tokens)} (None means allow all tokens)")
        
        print("🎉 PicardDecoder test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_picard_decoder()
