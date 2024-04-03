from bs4 import BeautifulSoup, Tag
from copy import deepcopy, copy
from models.model import Model
from typing import List

class InputChunker:
    def __init__(self, model: Model) -> None:
        self.model = model  # model object for GPT models or other models (HuggingFace)

    def __preprocess_markdown(self, md_string: str) -> str:
        """
        Preprocess the xml string by converting to a BeautifulSoup object and removing the styling tags.

        Args:
        xml_string: string

        Returns:
        soup: BeautifulSoup object
        """
        if remove_tags is None:
            remove_tags = ["bold", "italic", "underline", "sup", "sub", "xref"]
        soup = self.__convert_xml_string_to_soup(xml_string)
        soup = self.__remove_style_tags(soup, remove_tags)

        return soup

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

    def __combine_xml_chunks(self, chunks_list: List[BeautifulSoup], max_tokens: int) -> List[str]:
        """
        Combine the chunks based on the max tokens.

        Args:
        chunks_list: list
        max_tokens: integer

        Returns:
        final_chunks: list
        """
        final_chunks = []
        current_chunk = ""
        current_length = 0

        for soup in chunks_list:
            soup_length = self.count_tokens(str(soup))
            # If the soup is too long, print ERROR.
            # This should not happen ideally, but if it does, we should know about it.
            if soup_length > max_tokens:
                # print(str(soup))
                print(f"ERROR - chunk to combine is too long: {soup_length} tokens")
                continue # just skipping since there isn't a very clear way to deal with this currently
            # If adding this soup would exceed max_length, finish the current chunk
            if current_length + soup_length > max_tokens:
                if current_length > 0:  # Avoid adding empty chunks
                    chunk_to_add = {
                        "chunk": current_chunk,
                        "token_size": current_length
                    }
                    final_chunks.append(chunk_to_add)
                    current_chunk = ""  # Reset the current chunk
                    current_length = 0  # Reset the current length
                # Start a new chunk with the current soup
                current_chunk += str(soup)
                current_length += soup_length
            else:
                # If adding this soup wouldn't exceed max_length, add it to the current chunk
                current_chunk += str(soup)
                current_length += soup_length

        # After the loop, add the current_chunk if it's not empty
        if current_length > 0:
            chunk_to_add = {
                "chunk": current_chunk,
                "token_size": current_length
            }
            final_chunks.append(chunk_to_add)

        return final_chunks

    def get_chunked_input(self, md_strng: str, max_chunk_token_size: int) -> List[str]:
        """
        Split a text into chunks of ~max_num_tokens tokens
        
        Args:
        md_strng: string
        max_chunk_token_size: integer
        
        Returns:
        chunked_input: A list of text chunks
        """
        # soup = self.__preprocess_xml(xml_string)
        # xml_chunks_list = self.__create_xml_chunks(soup, max_chunk_token_size)
        # condensed_chunks_list = self.__combine_xml_chunks(xml_chunks_list, max_chunk_token_size)
        # return condensed_chunks_list
