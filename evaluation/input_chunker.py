from models.model import Model
from typing import List
from numerizer import numerize
from nltk.tokenize import sent_tokenize

# nltk.download("punkt") # Uncomment this line if you haven't downloaded the punkt tokenizer before

class InputChunker:
    def __init__(self, model: Model) -> None:
        self.model = model  # model object for GPT models or other models (HuggingFace)

    def __contains_digits(self, string_to_check: str) -> bool:
        """
        Check if a string contains digits.

        Args:
        string_to_check: string

        Returns:
        boolean
        """
        return any(char.isdigit() for char in string_to_check)
    
    def __preprocess_markdown(self, md_string: str) -> str:
        """
        Preprocess the markdown (md) string by converting number word to digits.
        Then, remove the non-numerical text from the markdown string.
        Tables do not get removed.

        Args:
        xml_string: string

        Returns:
        processed_string: string with number words converted to digits
        """
        # Convert number words to digits
        try:
            numerized_text = numerize(md_string)
        except:
            print(f"error in numerizing {md_string}")
            numerized_text = md_string

        # Remove non-numerical text from the markdown string
        processed_string = ""
        special_string = "::::" # special string to split the regular text and tables
        split_text = numerized_text.split(special_string)
        for partial_text in split_text:
            if "table-wrap" in partial_text:
                new_line = special_string + partial_text + special_string
                processed_string += new_line
            else:
                # get list of setences
                sentences = sent_tokenize(partial_text)
                new_lines = []
                for sent in sentences:
                    if self.__contains_digits(sent):
                        new_lines.append(sent)
                joined_content = " ".join(new_lines)
                processed_string += joined_content

        return processed_string

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
        text: string
        
        Returns:
        token_count: integer
        """
        encoded = self.model.encode_text(text)
        encoded_length = len(encoded)
        return encoded_length

    def __chunk_md_string(self, md_string: str, max_tokens: int) -> List[str]: # could technically do this with pre-processing
        """
        chunk the markdown string into chunks of ~max_tokens tokens

        Args:
        md_string: list
        max_tokens: integer

        Returns:
        final_chunks: list
        """
        final_chunks = []
        current_chunk = ""
        current_length = 0

        special_string = "::::" # special string to split the regular text and tables
        split_text = md_string.split(special_string)

        for partial_text in split_text:
            if "table-wrap" in partial_text:
                table = special_string + partial_text + special_string
                table_token_count = self.count_tokens(table)
                if current_length + table_token_count > max_tokens:
                    if current_length > 0:  # Avoid adding empty chunks
                        chunk_to_add = {
                            "chunk": current_chunk,
                            "token_size": current_length
                        }
                        final_chunks.append(chunk_to_add)
                        current_chunk = ""  # Reset the current chunk
                        current_length = 0  # Reset the current length
                    # Start a new chunk with the current table
                    current_chunk += str(table)
                    current_length += table_token_count
                else:
                    # If adding this table wouldn't exceed max_tokens, add it to the current chunk
                    current_chunk += str(table)
                    current_length += table_token_count
            else:
                sentences = sent_tokenize(partial_text)
                for sentence in sentences:
                    sentence_token_count = self.count_tokens(sentence)
                    if current_length + sentence_token_count > max_tokens:
                        if current_length > 0:  # Avoid adding empty chunks
                            chunk_to_add = {
                                "chunk": current_chunk,
                                "token_size": current_length
                            }
                            final_chunks.append(chunk_to_add)
                            current_chunk = ""  # Reset the current chunk
                            current_length = 0  # Reset the current length
                        # Start a new chunk with the current sentence
                        current_chunk += str(sentence)
                        current_length += sentence_token_count
                    else:
                        # If adding this chunk wouldn't exceed max_tokens, add it to the current chunk
                        current_chunk += str(sentence)
                        current_length += sentence_token_count

        # After the loop, add the current_chunk if it's not empty
        if current_length > 0:
            chunk_to_add = {
                "chunk": current_chunk,
                "token_size": current_length
            }
            final_chunks.append(chunk_to_add)

        return final_chunks
    
    def get_chunked_input(self, md_string: str, max_chunk_token_size: int) -> List[str]:
        """
        Split a text into chunks of ~max_num_tokens tokens
        
        Args:
        md_string: string
        max_chunk_token_size: integer
        
        Returns:
        chunked_input: A list of text chunks
        """
        processed_md_string = self.__preprocess_markdown(md_string)
        chunks_list = self.__chunk_md_string(processed_md_string, max_chunk_token_size)
        return chunks_list
