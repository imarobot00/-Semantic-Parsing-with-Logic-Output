from picard import sql_state, tokenizer_utils
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
            decoded_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
            # Trim to partial SQL
            trimmed_sql = decoded_text.split("<pad>")[-1].strip()
            valid_next_tokens = sql_state.get_valid_next_tokens(
                sql=trimmed_sql,
                db_id=db_id,
                schemas=self.schemas,
                tokenizer=self.tokenizer,
                database_path=self.db_path
            )
            return valid_next_tokens
        except Exception as e:
            print("Picard step error:", e)
            return None
