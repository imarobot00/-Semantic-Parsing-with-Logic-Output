from picard.sql_state import get_valid_next_tokens
from transformers import PreTrainedTokenizer


class PicardDecoder:
    def __init__(self, tokenizer: PreTrainedTokenizer, db_path: str, schemas: dict):
        self.tokenizer = tokenizer
        self.schemas = schemas
        self.db_path = db_path

    def step(self, input_ids, decoded_ids, db_id):
        """
        Given the current input_ids and decoded_ids, return allowed token ids.
        """
        try:
            # Decode current output (partial SQL)
            decoded_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
            trimmed_sql = decoded_text.split("<pad>")[-1].strip()

            # Get valid next tokens using Picard grammar + schema rules
            valid_next_tokens = get_valid_next_tokens(
                sql=trimmed_sql,
                db_id=db_id,
                schemas=self.schemas,
                tokenizer=self.tokenizer,
                database_path=self.db_path
            )
            return valid_next_tokens

        except Exception as e:
            print("⚠️ Picard constraint failed:", e)
            return None
