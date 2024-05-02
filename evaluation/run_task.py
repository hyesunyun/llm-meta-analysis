import argparse
import os

from models.gpt35 import GPT35
from models.gpt4 import GPT4
from models.biomistral import BioMistral
from models.gemma import Gemma
from models.mistral import Mistral
from models.pmc_llama import PMCLlama
from models.olmo import Olmo
from models.alpaca import Alpaca
from models.model import Model

from input_chunker import InputChunker

from utils import (
    format_example_with_prompt_template, 
    load_json_file,
    get_md_content_by_pmcid,
    save_json_file,
    save_dataset_to_json,
    save_dataset_to_csv,
    convert_character_to_string_outcome_type
)
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from templates import DatasetTemplates
import random

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")
XML_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data", "no_attributes_xml_files")
MD_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data", "no_attributes_markdown_files")

class MetaAnalysisTaskRunner:

    def __init__(self, model_name: str, task: str, split: str, output_path: str, is_test: bool=False, prompt_name: Optional[str]=None, input_path: Optional[str]=None, pmc_files_path: Optional[str]=None) -> None:
        '''
        This class runs the meta analysis task for the given model, task, and split

        :param model_name: name of the model to use
        :param task: task to run
        :param split: split of the dataset to run (dev or test)
        :param output_path: path to save the output data
        :param is_test: whether to run the task with only 10 instances for debugging purposes. default is False
        :param prompt_name: name of the prompt template to use (Optional, default is the first prompt template for the given task)
        :param input_path: path to the input file (Optional, default is the annotated_rct_dataset.json file in the data folder)
        :param pmc_files_path: path to the folder where the PMC files are stored

        :return None
        '''
        self.model_name = model_name
        self.task = task
        self.split = split
        self.prompt_name = prompt_name
        self.output_path = output_path
        self.is_test = is_test
        # the default input path is the annotated_rct_dataset.json file in the data folder
        # this file is the evaluation dataset
        self.input_path = input_path if input_path is not None else os.path.join(DATA_FOLDER_PATH, "annotated_rct_dataset.json")
        # the default input path is the no_attributes_markdown_files folder
        self.pmc_files_path = pmc_files_path if pmc_files_path is not None else MD_FOLDER_PATH

        self.prompt_template = None
        self.dataset = None
        self.model = None
        self.max_new_tokens = self.__get_max_new_tokens()

        self.__load_prompt_template()
        self.__load_dataset()
        self.__load_model()

    def __load_prompt_template(self) -> str:
        """
        This method loads the prompt template for the given task and model

        :return string of the full prompt template
        """
        if "gpt" in self.model_name:
            self.prompt_template = self.task + "/gpt"
        elif "pmc-llama" in self.model_name: 
            self.prompt_template = self.task + "/pmc-llama"
        elif self.model_name == "mistral7B":
            self.prompt_template = self.task + "/mistral"
        elif self.model_name == "biomistral":
            self.prompt_template = self.task + "/biomistral"
        elif "gemma" in self.model_name:
            self.prompt_template = self.task + "/gemma"
        elif "olmo" in self.model_name:
            self.prompt_template = self.task + "/olmo"
        elif self.model_name == "alpaca13B":
            self.prompt_template = self.task + "/alpaca"
        else:
            self.prompt_template = self.task # default. this should never really happen
    
    def __load_dataset(self) -> List[Dict]:
        """
        This method loads the dataset (test split)

        :return dataset as a list of dictionaries
        """
        dataset = load_json_file(self.input_path)

        # get the correct split to run the task if split is provided
        if self.split == "dev":
            dataset = [example for example in dataset if example["split"] == "DEV"]
        elif self.split == "test":
            # filter out dataset to only test (evaluation) split
            dataset = [example for example in dataset if example["split"] == "TEST"]

        # get only examples for the given task
        if self.task == "binary_outcomes":
            dataset = [example for example in dataset if example["outcome_type"] == "binary"]
        elif self.task == "continuous_outcomes":
            dataset = [example for example in dataset if example["outcome_type"] == "continuous"]

        # if test, only get 10 random examples
        if self.is_test:
            random.shuffle(dataset)
            dataset = dataset[:10] if len(dataset) > 10 else dataset

        # Add md content to each example
        for example in dataset:
            pmcid = example["pmcid"]
            md_content = get_md_content_by_pmcid(self.pmc_files_path, pmcid)
            md_item = {"abstract_and_results": md_content}
            example.update(md_item)

        # shuffle the dataset since it is ordered by pmcid
        random.seed(42) # set seed for reproducibility
        random.shuffle(dataset)
        self.dataset = dataset

    def __load_model(self) -> Model:
        """
        This method loads the model requested for the task

        :return Model object
        """
        model_class_mapping = {"gpt35": GPT35, "gpt4": GPT4, "mistral7B": Mistral, "biomistral": BioMistral, "pmc-llama": PMCLlama, "gemma7B": Gemma, "olmo7B": Olmo, "alpaca13B": Alpaca}
        model_class = model_class_mapping[self.model_name]
        self.model = model_class()

    def __get_max_new_tokens(self) -> int:
        """
        This method returns the maximum number of new tokens to add to the model for the given task

        :param task: task to get the max new tokens for

        :return maximum number of new tokens
        """
        max_new_tokens = {
            "outcome_type": 5,
            "binary_outcomes": 50,
            "continuous_outcomes": 70,
        }
        return max_new_tokens[self.task]

    def run_task(self) -> Tuple[str, str]:
        '''
        This method runs the task for the given model, task, and split

        :return paths to the output file (json and csv) as a tuple
        '''

        # check ift there is any data
        if len(self.dataset) == 0:
            print("There is no data to run.")
            return None, None

        # load dataset prompt templates
        prompts = DatasetTemplates(self.prompt_template)

        # if prompt name is given, apply the prompt template. if not, use all 
        if self.prompt_name is None:
            all_prompt_templates = prompts.all_template_names
            prompt_template_name = all_prompt_templates[0]
        else:
            prompt_template_name = self.prompt_name
        
        prompt = prompts[prompt_template_name]

        # CODE WITH CHUNKING
        # format the dataset with the prompt template
        dataset = [format_example_with_prompt_template(example, prompt) for example in tqdm(self.dataset)]

        # for outcome_type, we don't need chunking since we don't add any abstract/results text to prompt
        # we can also fit everything into gpt4-turbo
        if self.task == "outcome_type" or self.model_name == "gpt4":
            # run the task using specified model
            results = []
            pbar = tqdm(dataset)
            for _, example in enumerate(pbar):
                output = self.model.generate_output(example["input"], max_new_tokens=self.max_new_tokens)
                example["output"] = output
                results.append(example)
        else: # for binary_outcomes or continuous_outcomes not using gpt4, we may need to chunk the input
            # instantiate input chunker
            input_chunker = InputChunker(self.model)

            # run the task using specified model
            results = []
            pbar = tqdm(dataset)
            for _, example in enumerate(pbar):
                input_token_count = input_chunker.count_tokens(example["input"])

                if input_token_count <= self.model.get_context_length(): # if the model can handle the tokens, just do as normal
                    output = self.model.generate_output(example["input"], max_new_tokens=self.max_new_tokens)
                    example["chunk_num_tokens"] = []
                    example["output"] = output
                    example["is_chunked"] = False
                    results.append(example)
                else: # if the model cannot handle the tokens, chunk the input
                    prompt_approx_tokens = 450 if self.model_name == "pmc-llama" or self.model_name == "alpaca13B" else 300 # PMC LLAMA & Alapaca prompts tend to be longer. Also model seems to be more sensitive to token size
                    max_chunk_tokens = self.model.get_context_length() - prompt_approx_tokens - self.max_new_tokens # account for the actual prompt, 300 as approx num of tokens of prompt template and also tokens to generate
                    chunks = input_chunker.get_chunked_input(example["abstract_and_results"], max_chunk_tokens)
                    chunked_examples = []
                    for chunk in chunks:
                        chunked_example = example.copy()
                        chunked_example["abstract_and_results"] = chunk["chunk"]
                        chunked_example["model_chunk_token_size"] = chunk["token_size"]
                        chunked_examples.append(chunked_example)
                    # format the chunks with the prompt template
                    formatted_chunked_examples = [format_example_with_prompt_template(example, prompt) for example in chunked_examples]

                    concatenated_output = ""
                    chunk_num_tokens_list = []
                    for input_chunk in formatted_chunked_examples:
                        chunk_num_tokens_list.append(input_chunk["model_chunk_token_size"])
                        output = self.model.generate_output(input_chunk["input"], max_new_tokens=self.max_new_tokens)
                        concatenated_output = concatenated_output + output + "\n---\n"
                    example["chunk_num_tokens"] = chunk_num_tokens_list
                    example["output"] = concatenated_output
                    example["is_chunked"] = True
                    results.append(example)

        # saving results to file
        print(f"Saving outputs for task - {self.task}; prompt - {prompt.get_name()}; model - {self.model_name} to csv and json")

        keys_to_drop = [
            "abstract_and_results", 
            "input"
        ]
        # convert into json
        json_file_path = f"{self.output_path}/{self.model_name}_{self.task}_{self.split}_output.json"
        save_dataset_to_json(dataset, json_file_path, keys_to_drop)

        # convert into csv
        csv_file_path = f"{self.output_path}/{self.model_name}_{self.task}_{self.split}_output.csv"
        save_dataset_to_csv(dataset, csv_file_path, keys_to_drop)

        return json_file_path, csv_file_path

def run_end_to_end_task(model: str, split: str, input_path:str, output_path: str, pmc_files_path: str, is_test: bool) -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    '''
    This method runs the end-to-end task for the given model and split
    
    :param model: name of the model to use
    :param split: split of the dataset to run (dev or test)
    :param input_path: path to the input file
    :param output_path: path to save the output data
    :param pmc_files_path: path to the folder where the PMC files are stored
    :param is_test: whether to run the task with only 10 instances for debugging purposes
    
    :return paths to the output files (json and csv) for outcome types and binary and continuous outcomes as a tuple
            output types: (json, csv) and binary outcomes: (json, csv) and continuous outcomes: (json, csv)
    '''
    # First, call the outcome type task
    outcome_type_task_runner = MetaAnalysisTaskRunner(model, "outcome_type", split, output_path, is_test, None, input_path, pmc_files_path)
    outcome_type_json_file_path, outcome_type_csv_file_path = outcome_type_task_runner.run_task()
    # Do some post-processing to pass the output to the next task as input
    outcome_type_outputs = load_json_file(outcome_type_json_file_path)
    for example in outcome_type_outputs:
        original_outcome_type = example["outcome_type"]
        model_output = example["output"]
        output_outcome_type = convert_character_to_string_outcome_type(model_output)
        new_item = {"outcome_type_reference": original_outcome_type, "outcome_type": output_outcome_type}
        example.update(new_item)
    save_json_file(outcome_type_json_file_path, outcome_type_outputs)

    # Second, call binary_outcomes task using the output from the outcome_type task
    binary_outcomes_task_runner = MetaAnalysisTaskRunner(model, "binary_outcomes", split, output_path, is_test, None, outcome_type_json_file_path, pmc_files_path)
    binary_outcomes_json_file_path, binary_outcomes_csv_file_path = binary_outcomes_task_runner.run_task()

    # Third, call continuous_outcomes task using the output from the outcome_type task
    continuous_outcomes_task_runner = MetaAnalysisTaskRunner(model, "continuous_outcomes", split, output_path, is_test, None, outcome_type_json_file_path, pmc_files_path)
    continuous_outcomes_json_file_path, continuous_outcomes_csv_file_path = continuous_outcomes_task_runner.run_task()

    return (outcome_type_json_file_path, outcome_type_csv_file_path), (binary_outcomes_json_file_path, binary_outcomes_csv_file_path), (continuous_outcomes_json_file_path, continuous_outcomes_csv_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Clinical Trials Meta Analysis Task")

    parser.add_argument("--model", default="gpt35", choices=["gpt35", "gpt4", "mistral7B", "biomistral", "pmc-llama", "gemma7B", "olmo7B", "alpaca13B"], help="what model to run", required=True)
    parser.add_argument("--task", default="outcome_type", choices=['outcome_type', 'binary_outcomes', 'continuous_outcomes', 'end_to_end'], help="type of task to run", required=True)
    parser.add_argument("--split", default=None, choices=["test", "dev"], help="which split of the default dataset to run. Required if input_path is not specified and overrides input_path.")
    parser.add_argument("--prompt", default=None, help="specific prompt to run. if no specific prompt is given, the first prompt for the given task is run. OPTIONAL")
    parser.add_argument("--input_path", default=None, help="directory of where the input data is stored. this is required if split is not specified.")
    parser.add_argument("--output_path", default="./output", help="directory of where the outputs/results should be saved.")
    parser.add_argument("--pmc_files_path", default=None, help="directory of where the PMC files are stored. Default is the no_attributes_markdown_files folder. OPTIONAL")
    # do --no-test for explicit False
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, help="used for debugging purposes. test will only run random 10 instances from the dataset.")
    
    args = parser.parse_args()

    model = args.model
    task = args.task
    split = args.split
    prompt_name = args.prompt
    input_path = args.input_path
    output_path = args.output_path
    pmc_files_path = args.pmc_files_path
    is_test = args.test

    print("Arguments Provided for the Clinical Trials Meta Analysis Task Runner:")
    print(f"Model:        {model}")
    print(f"Task:         {task}")
    print(f"Split:        {split}")
    print(f"Prompt Name:  {prompt_name}")
    print(f"Input Path:   {input_path}")
    print(f"Output Path:  {output_path}")
    print(f"PMC Files:    {pmc_files_path}")
    print(f"Is Test:      {is_test}")
    print()

    if split is None and input_path is None:
        raise ValueError("--split or --input_path must be specified.")
    if split is not None and input_path is not None:
        print("Both --split and --input_path are specified. Using split.")
        input_path = None

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    if task == "end_to_end":
        outcome_type_task_files, binary_outcomes_task_files, continuous_outcomes_task_files = run_end_to_end_task(model, task, split, output_path, is_test, prompt_name, input_path, pmc_files_path)
        print(f"Outcome Type task outputs saved to {outcome_type_task_files[0]} and {outcome_type_task_files[1]}")
        print(f"Binary Outcomes task outputs saved to {binary_outcomes_task_files[0]} and {binary_outcomes_task_files[1]}")
        print(f"Continuous Outcomes task outputs saved to {continuous_outcomes_task_files[0]} and {continuous_outcomes_task_files[1]}")
    else:
        task_runner = MetaAnalysisTaskRunner(model, task, split, output_path, is_test, prompt_name, input_path, pmc_files_path)
        json_file_path, csv_file_path = task_runner.run_task()
        print(f"Task outputs saved to {json_file_path} and {csv_file_path}")
    
    