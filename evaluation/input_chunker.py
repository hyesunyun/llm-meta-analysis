from bs4 import BeautifulSoup, Tag
from copy import deepcopy
from templates import DatasetTemplates, Template
from utils import format_example_with_prompt_template
from models.model import Model
from typing import Tuple

# This class is responsible for chunking the input based on the max tokens.
# Majority of the code was implemented by David Pogrebitskiy (@pogrebitskiy)
class InputChunker:
    def __init__(self, model_name: str, model: Model) -> None:
        self.model_name = model_name # the name of the model (parameter for run_task.py)
        self.model = model # model object for GPT models or other models (HuggingFace)
        self.prompt_template = None
        
        self.__load_prompt_template() # loads the actual prompt template for the model

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
        return len(self.model.encode_text(text))
    
    def __load_prompt_template(self) -> Template:
        """
        This method loads the prompt template for the given model class name.

        :return string of the full prompt template
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
            prompt_template = "chunking/" # default. this should never really happen

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
        answer = answer.replace("The answer is ", "").replace(".", "").replace("(", "").replace(")", "") # remove any parens, periods, and other known common, extra texts
        # remove any unnecessary text output by finding the first non-space character
        for char in answer:
            if not char.isspace():
                answer = char
                break
        string_answer = character_to_string_mapping[answer] if answer in character_to_string_mapping else 'y' # not sure how to handle situations like this
        if string_answer == 'n':
            return False
        elif string_answer == 'y':
            return True
        else:
            raise ValueError("Model answer is not valid.")
    
    def __is_relevant(self, text: str, ico_dict: dict) -> bool:
        """
        Check if the text is relevant.
        
        Args:
        text: string
        
        Returns:
        is_relevant: boolean
        """
        example = ico_dict
        example["chunk"] = text
        example = format_example_with_prompt_template(ico_dict, self.prompt_template)
        print(example["input"])
        model_output = self.model.generate_output(example["input"], 3)
        model_output = model_output.strip()

        return self.__convert_output_to_boolean(model_output)
    
    def __chunk_xml(self, xml_soup_element: BeautifulSoup, ico_dict: dict, max_tokens: int) -> Tuple[list, int]:
        """
        Chunk the xml soup element based on the max tokens.

        Args:
        xml_soup_element: BeautifulSoup object
        ico_dict: dictionary
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

            is_p_tag = chunk.name == 'p'
            is_table = isinstance(chunk, Tag) and chunk.name == 'table-wrap'

            if (is_table and is_relevant) or (is_p_tag and is_relevant):
                keep_chunks.append(chunk)

            elif self.count_tokens(str(chunk)) >= max_tokens and is_relevant and not is_table:
                # Chunk it further, recursively
                keep_chunks.extend(self.__chunk_xml(chunk, max_tokens))

            elif self.count_tokens(str(chunk)) < max_tokens and is_relevant:
                # if the chunk is too small and the condition is true, keep it
                keep_chunks.append(chunk)
            
            else:
                # If the chunk is not relevant, don't keep it
                pass

        for child in xml_soup_element.contents:
            process_chunk()

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
                if current_length > 0: # Avoid adding empty chunks
                    final_chunks.append(current_chunk)
                # Start a new chunk with the current soup
                current_chunk = soup
                current_length = soup_length
            else:
                # If adding this soup wouldn't exceed max_length, add it to the current chunk
                current_chunk.append(soup)
                current_length += soup_length

        # After the loop, add the last chunk if it's not empty
        if current_length > 0:
            final_chunks.append(current_chunk)

        return final_chunks

    
    def get_chunked_input(self, xml_string: str, ico_dict: dict, max_tokens: int) -> Tuple[list, int]:
        """
        Get the chunked input based on the max tokens. The only public method.
        
        Args:
        xml_string: string
        max_tokens: integer
        
        Returns:
        chunked_input: list
        num_model_calls: integer
        """
        soup = self.__convert_xml_string_to_soup(xml_string)
        chunks_list, num_model_calls = self.__chunk_xml(soup, ico_dict, max_tokens)
        condensed_chunks_list = self.__combine_chunks(chunks_list, max_tokens)
        return condensed_chunks_list, num_model_calls
        