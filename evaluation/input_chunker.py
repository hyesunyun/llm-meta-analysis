from bs4 import BeautifulSoup, Tag
from copy import deepcopy
from templates import DatasetTemplates, Template
from utils import format_example_with_prompt_template
from models.model import Model

# This class is responsible for chunking the input based on the max tokens.
# Majority of the code was implemented by David Pogrebitskiy (@pogrebitskiy)
class InputChunker:
    def __init__(self, model: Model) -> None:
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
        return len(self.tokenizer.encode_text(text))
    
    def __load_prompt_template(self) -> Template:
        """
        This method loads the prompt template for the given model class name.

        :return string of the full prompt template
        """
        model_class_name = self.model.__class__.__name__
        print(model_class_name)
        if "gpt" in model_class_name:
            prompt_template = "chunking/gpt"
        elif "pmc-llama" in model_class_name: 
            prompt_template = "chunking/pmc-llama"
        elif model_class_name == "mistral7B":
            prompt_template = "chunking/mistral"
        elif model_class_name == "biomistral":
            prompt_template = "chunking/biomistral"
        elif "gemma" in model_class_name:
            prompt_template = "chunking/gemma"
        elif "olmo" in model_class_name:
            prompt_template = "chunking/olmo"
        else:
            prompt_template = "chunking/" # default. this should never really happen

        prompts = DatasetTemplates(prompt_template)
        prompt_template_name = prompts.all_template_names[0]
        prompt = prompts[prompt_template_name]

        self.prompt = prompt

    def __convert_output_to_boolean(answer: str) -> bool:
        """
        This method converts the answer from string to boolean.

        :param answer: answer as character

        :return answer as string
        """
        character_to_string_mapping = {"A": "n", "B": "y"}
        answer = answer.replace("(", "").replace(")", "") # remove any parens
        # remove any unnecessary text output by finding the first non-space character
        for char in answer:
            if not char.isspace():
                answer = char
                break
        string_answer = character_to_string_mapping[answer]
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
        example["input"] = format_example_with_prompt_template(ico_dict, self.prompt)

        model_output = self.model.generate_output(example["input"], 1)
        model_output = model_output.strip()

        return self.__convert_output_to_boolean(model_output)
    
    def __chunk_xml(self, xml_soup_element: BeautifulSoup, ico_dict: dict, max_tokens: int) -> list:
        """
        Chunk the xml soup element based on the max tokens.

        Args:
        xml_soup_element: BeautifulSoup object
        ico_dict: dictionary
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
            is_relevant = self.__is_relevant(chunk, ico_dict)

            is_p_tag = chunk.name == 'p'
            is_table = isinstance(chunk, Tag) and chunk.name == 'table-wrap'

            if (is_table and is_relevant) or (is_p_tag and is_relevant):
                keep_chunks.append(chunk)

            elif self.__count_tokens(str(chunk)) >= max_tokens and is_relevant and not is_table:
                # Chunk it further, recursively
                keep_chunks.extend(self.__chunk_xml(chunk, max_tokens))

            elif self.__count_tokens(str(chunk)) < max_tokens and is_relevant:
                # if the chunk is too small and the condition is true, keep it
                keep_chunks.append(chunk)
            
            else:
                # If the chunk is not relevant, don't keep it
                pass

        for child in xml_soup_element.contents:
            process_chunk(child)

        return keep_chunks
    
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
            soup_length = self.__count_tokens(str(soup))
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

    
    def get_chunked_input(self, xml_string: str, ico_dict: dict, max_tokens: int) -> list:
        """
        Get the chunked input based on the max tokens. The only public method.
        
        Args:
        xml_string: string
        max_tokens: integer
        
        Returns:
        chunked_input: list
        """
        soup = self.__convert_xml_string_to_soup(xml_string)
        chunks_list = self.__chunk_xml(soup, ico_dict, max_tokens)
        condensed_chunks_list = self.__combine_chunks(chunks_list, max_tokens)
        return condensed_chunks_list
        