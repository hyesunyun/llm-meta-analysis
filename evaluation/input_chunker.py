from bs4 import BeautifulSoup, Tag
from copy import deepcopy, copy
from models.model import Model
from typing import List

# This class is responsible for chunking the input based on the max tokens.
# Majority of the code was implemented by David Pogrebitskiy (@pogrebitskiy)
class InputChunker:
    def __init__(self, model) -> None:
        self.model = model  # model object for GPT models or other models (HuggingFace)

    def __remove_style_tags(self, soup: BeautifulSoup, tags: list) -> BeautifulSoup:
        """
        Remove the style tags from the soup object.

        Args:
        soup: BeautifulSoup object
        tags: list

        Returns:
        soup: BeautifulSoup object
        """
        # Copy the soup and unwrap the styling tags specified in the list
        soup = deepcopy(soup)
        for tag in soup.find_all(tags):
            tag.unwrap()
        return soup

    def __preprocess_xml(self, xml_string: str, remove_tags: list = None) -> BeautifulSoup:
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

    def __remove_html_body(self, soup_object: BeautifulSoup) -> BeautifulSoup:
        """
        Remove the html and body tags from the soup object. 
        This is necessary because the lxml parser adds these tags automatically.
        
        Args:
        soup: BeautifulSoup object

        Returns:
        soup: BeautifulSoup object with html and body tags removed
        """
        html_tag = soup_object.html
        body_tag = soup_object.body

        # Unwrap the unnecessary tags that are added by lxml parser
        if html_tag is not None:
            html_tag.unwrap()
        if body_tag is not None:
            body_tag.unwrap()

        return soup_object

    def __convert_xml_string_to_soup(self, xml_string: str) -> BeautifulSoup:
        """
        Convert the xml string to a BeautifulSoup object.
        
        Args:
        xml_string: string
        
        Returns:
        soup: BeautifulSoup object
        """
        soup = BeautifulSoup(xml_string, "lxml")

        # Remove the html and body tags
        return self.__remove_html_body(soup)

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

    def __split_table(self, table: BeautifulSoup) -> List[BeautifulSoup]:
        """
        Extract the header and footer, spit the body table rows in half, and return the two tables (first and second halves).

        Args:
        table: BeautifulSoup object

        Returns:
        list of BeautifulSoup objects
        """

        # Copy the input table
        table_copy = deepcopy(table)

        # Keep track of the header
        header = BeautifulSoup("", 'lxml')
        [header.append(copy(tag)) for tag in table_copy.find_all(('label', 'caption'))]
        thead = table_copy.find('thead')
        if thead is not None:
            header.append(copy(thead))

        # Keep track of the footer
        footer = table_copy.find('table-wrap-foot')

        # Find all rows in the table
        all_rows = table_copy.find('tbody').find_all('tr', recursive=False)
        num_rows = len(all_rows)

        # Split the rows in half
        first_half = all_rows[:num_rows // 2]
        second_half = all_rows[num_rows // 2:]

        # Create the first table
        first_table = BeautifulSoup("", 'lxml')
        first_table.append(copy(header))
        first_tbody = first_table.new_tag('tbody')
        [first_tbody.append(copy(row)) for row in first_half]
        first_table.append(copy(first_tbody))
        if footer:
            first_table.append(copy(footer))
        first_table_wrap = first_table.new_tag('table-wrap')
        first_table_wrap.append(copy(first_table))

        # Create the second table
        second_table = BeautifulSoup("", 'lxml')
        second_table.append(copy(header))
        second_tbody = second_table.new_tag('tbody')
        [second_tbody.append(copy(row)) for row in second_half]
        second_table.append(copy(second_tbody))
        if footer:
            second_table.append(copy(footer))
        second_table_wrap = second_table.new_tag('table-wrap')
        second_table_wrap.append(copy(second_table))

        return [first_table_wrap, second_table_wrap]

    def __create_xml_chunks(self, xml_soup_element: BeautifulSoup, max_tokens: int) -> List[str]:
        """
        Recursively chunk the xml soup until each chunk is less than max_tokens.

        Args:
        xml_soup_element: BeautifulSoup object
        max_tokens: integer

        Returns:
        keep_chunks: list
        """
        keep_chunks = []

        def process_chunk(chunk: BeautifulSoup) -> None:
            """
            Process the chunk.

            Args:
            chunk: BeautifulSoup object

            Returns:
            None
            """
            chunk = deepcopy(chunk)
            chunk_token_size = self.count_tokens(str(chunk))
            is_table = chunk.name == 'table-wrap'

            if chunk_token_size > max_tokens:
                if is_table:
                    # If the chunk is a table, but it's too large, split it in half and add the two tables to the list
                    keep_chunks.extend(self.__split_table(chunk))
                else:
                    # If the chunk is too large, chunk it further
                    keep_chunks.extend(self.__create_xml_chunks(chunk, max_tokens))

            # If the chunk is small enough, add it to the list
            else:
                keep_chunks.append(copy(chunk))

        # Iterate through the children of the XML element
        for child in xml_soup_element.contents:
            if isinstance(child, Tag):
                # Process the chunk
                process_chunk(copy(child))
            else:
                keep_chunks.append(copy(child))

        # Return the list of chunks
        return keep_chunks

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
                continue
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

    def get_chunked_input(self, xml_string: str, max_chunk_token_size: int) -> List[str]:
        """
        Split a text into chunks of ~max_num_tokens tokens, based on xml tag boundaries.
        
        Args:
        xml_string: string
        max_chunk_token_size: integer
        
        Returns:
        chunked_input: A list of text chunks
        """
        soup = self.__preprocess_xml(xml_string)
        xml_chunks_list = self.__create_xml_chunks(soup, max_chunk_token_size)
        condensed_chunks_list = self.__combine_xml_chunks(xml_chunks_list, max_chunk_token_size)
        return condensed_chunks_list
