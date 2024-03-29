from bs4 import BeautifulSoup, Tag, NavigableString
from copy import deepcopy, copy
from templates import DatasetTemplates, Template
from utils import format_example_with_prompt_template
from models.model import Model
from typing import Tuple


# This class is responsible for chunking the input based on the max tokens.
# Majority of the code was implemented by David Pogrebitskiy (@pogrebitskiy)
class InputChunker:
    def __init__(self, model_name: str, model: Model) -> None:
        self.model_name = model_name  # the name of the model (parameter for run_task.py)
        self.model = model  # model object for GPT models or other models (HuggingFace)
        self.prompt_template = None
        self.min_chunk_tokens = 250  # minimum tokens to stop reducing the chunks

        self.__load_prompt_template()  # loads the actual prompt template for the model

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
            remove_tags = ["bold", "italic", "underline", "sup", "sub"]
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
        return len(self.model.encode_text(text))

    def __load_prompt_template(self):
        """
        This method loads the prompt template for the given model class name.

        Returns:
            None
        """
        if "gpt" in self.model_name:
            prompt_template = "chunking/gpt"
        elif "pmc-llama" == self.model_name:
            prompt_template = "chunking/pmc-llama"
        elif "mistral" in self.model_name:
            prompt_template = "chunking/mistral"
        elif "biomistral" == self.model_name:
            prompt_template = "chunking/biomistral"
        elif "gemma" in self.model_name:
            prompt_template = "chunking/gemma"
        elif "olmo" in self.model_name:
            prompt_template = "chunking/olmo"
        else:
            prompt_template = "chunking/"  # default. this should never really happen

        prompts = DatasetTemplates(prompt_template)
        all_prompt_templates = prompts.all_template_names
        prompt_template_name = all_prompt_templates[0]
        prompt = prompts[prompt_template_name]

        self.prompt_template = prompt

    def __convert_output_to_boolean(self, answer: str) -> bool:
        """
        This method converts the answer from string to boolean.

        :param answer: answer as character

        :return answer as string
        """
        print(answer)
        character_to_string_mapping = {"A": "n", "B": "y"}
        answer = answer.replace("The answer is ", "").replace(".", "").replace("(", "").replace(")",
                                                                                                "")  # remove any parens, periods, and other known common, extra texts
        # remove any unnecessary text output by finding the first non-space character
        for char in answer:
            if not char.isspace():
                answer = char
                break
        string_answer = character_to_string_mapping[
            answer] if answer in character_to_string_mapping else 'y'  # default to "yes" in the case that the model might find relevant data with further chunking (conservative approach)
        if string_answer == 'n':
            return False
        elif string_answer == 'y':
            return True
        else:
            raise ValueError("Model answer is not valid.")

    def __is_relevant(self, text: BeautifulSoup, ico_dict: dict) -> bool:
        """
        Check if the text is relevant.
        
        Args:
        text: BeautifulSoup object
        
        Returns:
        is_relevant: boolean
        """
        example = ico_dict
        example["chunk"] = text
        example = format_example_with_prompt_template(example, self.prompt_template)
        model_output = self.model.generate_output(example["input"], 3)
        model_output = model_output.strip()

        return self.__convert_output_to_boolean(model_output)

    def __split_table(self, table: Tag) -> list:
        """

        Extract the header and footer, spit the rows in half, and return the two tables.

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
        header.append(copy(table_copy.find('thead')))

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

    def __chunk_xml(self, xml_soup_element: Tag, ico_dict: dict, min_chunk_tokens: int, max_tokens: int) -> Tuple[
        list, int]:
        """
        Chunk the xml soup element based on the max tokens.

        Args:
        xml_soup_element: BeautifulSoup object
        ico_dict: dictionary
        min_chunk_tokens: integer
        max_tokens: integer

        Returns:
        keep_chunks: list
        num_model_calls: integer
        """
        keep_chunks = []
        num_model_calls = 0

        def process_chunk() -> None:
            """
            Process the chunk.

            Returns:
            None
            """
            nonlocal num_model_calls
            chunk = deepcopy(child)
            is_relevant = self.__is_relevant(chunk, ico_dict)
            num_model_calls += 1
            chunk_len = self.count_tokens(str(chunk))

            is_table = isinstance(chunk, Tag) and chunk.name == 'table-wrap'
            is_abstract = isinstance(chunk, Tag) and chunk.name == 'abstract'

            # If the chunk is a table and is relevant, but it's too large, split it in half and add the two tables to
            # the list
            if is_table and is_relevant and chunk_len > max_tokens:
                if is_table:
                    keep_chunks.extend(self.__split_table(chunk))

            # If the chunk is a table or abstract and is relevant, and not too big, append it to the list
            elif (is_table or is_abstract) and is_relevant and chunk_len <= max_tokens:
                keep_chunks.append(chunk)

            # If the chunk isn't smaller than the minimum chunk size, and is relevant, chunk it further
            elif chunk_len >= min_chunk_tokens and is_relevant:
                # Chunk it further, recursively
                keep_chunks.extend(self.__chunk_xml(chunk, ico_dict, min_chunk_tokens, max_tokens)[0])

            elif chunk_len < min_chunk_tokens and is_relevant:
                # if the chunk is too small and the condition is true, keep it
                keep_chunks.append(chunk)

            else:
                # If the chunk is not relevant, don't keep it
                pass

        for child in xml_soup_element.contents:
            # If the child is a tag, process the chunk
            if isinstance(child, Tag):
                process_chunk()
            # If it's anything else, append it and don't chunk further
            else:
                keep_chunks.append(xml_soup_element)
                continue

        return keep_chunks, num_model_calls

    def __combine_chunks(self, chunks_list: list, max_tokens: int) -> list:
        """
        Combine the chunks based on the max tokens.

        Args:
        chunks_list: list
        max_tokens: integer

        Returns:
        final_chunks: list
        """
        final_chunks = []
        current_chunk = BeautifulSoup("", 'lxml')
        current_length = 0

        for soup in chunks_list:
            soup_length = self.count_tokens(str(soup))
            # If adding this soup would exceed max_length, finish the current chunk
            if current_length + soup_length > max_tokens:
                if current_length > 0:  # Avoid adding empty chunks
                    final_chunks.append(copy(current_chunk))
                # Start a new chunk with the current soup
                current_chunk = copy(soup)
                current_length = soup_length
            else:
                # If adding this soup wouldn't exceed max_length, add it to the current chunk
                current_chunk.append(copy(soup))
                current_length += soup_length

        # After the loop, add the last chunk if it's not empty
        if current_length > 0:
            final_chunks.append(copy(current_chunk))

        return final_chunks

    def __postprocess_chunks(self, chunks_list: list) -> list:
        """
        Postprocess the chunks by stripping all tags that aren't within a table-wrap

        Args:
        chunks_list: list

        Returns:
        final_chunks: list
        """

        final_chunks = []

        for chunk in chunks_list:
            # If the chunk is a table-wrap, keep it as is
            if chunk.name == "table-wrap":
                final_chunks.append(copy(chunk))
            # If the chunk is an abstract, remove all tags then wrap it in an abstract tag
            elif chunk.name == "abstract":
                abstract_soup = BeautifulSoup('', 'lxml')
                abstract_tag = abstract_soup.new_tag('abstract')
                abstract_tag.string = chunk.get_text(' ')
                abstract_soup.append(abstract_tag)
            # Otherwise, remove all tags and keep the text
            else:
                final_chunks.append(copy(chunk.get_text(' ')))

        return final_chunks

    def get_chunked_input(self, xml_string: str, ico_dict: dict, max_tokens: int) -> Tuple[list, int]:
        """
        Get the chunked input based on the max tokens. The only public method.
        
        Args:
        xml_string: string
        max_tokens: integer
        
        Returns:
        chunked_input: list
        ico_dict: dictionary
        num_model_calls: integer
        """
        soup = self.__preprocess_xml(xml_string)
        chunks_list, num_model_calls = self.__chunk_xml(soup, ico_dict, self.min_chunk_tokens, max_tokens)
        chunks_list = self.__postprocess_chunks(chunks_list)
        condensed_chunks_list = self.__combine_chunks(chunks_list, max_tokens)
        return condensed_chunks_list, num_model_calls
