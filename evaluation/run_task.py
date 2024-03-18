import argparse
import os

from models.gpt35 import GPT35
from models.gpt4 import GPT4
from models.model import Model

from utils import (
    format_example_with_prompt_template, 
    load_json_file, 
    get_xml_content_by_pmcid,
    save_json_file,
    save_dataset_to_json,
    save_dataset_to_csv,
    convert_character_to_string_outcome_type
)
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from templates import DatasetTemplates
from datetime import datetime
import random

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")

class MetaAnalysisTaskRunner:

    def __init__(self, model_name: str, task: str, split: str, output_path: str, is_test: bool=False, prompt_name: Optional[str]=None, input_path: Optional[str]=None) -> None:
        '''
        This class runs the meta analysis task for the given model, task, and split

        :param model_name: name of the model to use
        :param task: task to run
        :param split: split of the dataset to run (dev or test)
        :param output_path: path to save the output data
        :param is_test: whether to run the task with only 10 instances for debugging purposes
        :param prompt_name: name of the prompt template to use (Optional, default is the first prompt template for the given task)
        :param input_path: path to the input file (Optional, default is the annotated_rct_dataset.json file in the data folder)

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

        self.prompt_template = None
        self.dataset = None
        self.model = None
        self.max_new_tokens = None

        self.__load_prompt_template()
        self.__load_dataset()
        self.__load_model()
        self.__get_max_new_tokens()

    def __load_prompt_template(self) -> str:
        """
        This method loads the prompt template for the given task

        :return string of the full prompt template
        """
        # TODO: this can be deleted if we are not going to use any models other than OpenAI ones
        # if we use some open source models, we should discriminate model too and add model name to path
        self.prompt_template = self.task
    
    def __load_dataset(self) -> List[Dict]:
        """
        This method loads the dataset (test split)

        :return dataset as a list of dictionaries
        """
        dataset = load_json_file(self.input_path)

        # get the correct split to run the task
        if self.split == "dev":
            dataset = [example for example in dataset if example["split"] == "DEV"]
        else:
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
            dataset = dataset[:10]

        # Add xml content to each example
        for example in dataset:
            pmcid = example["pmcid"]
            xml_content = get_xml_content_by_pmcid(pmcid)
            xml_item = {"abstract_and_results_xml": xml_content}
            example.update(xml_item)

        # shuffle the dataset since it is ordered by pmcid
        random.seed(42) # set seed for reproducibility
        random.shuffle(dataset)
        self.dataset = dataset

    def __load_model(self) -> Model:
        """
        This method loads the model requested for the task

        :return Model object
        """
        model_class_mapping = {"gpt35": GPT35, "gpt4": GPT4}
        model_class = model_class_mapping[self.model_name]
        self.model = model_class()

    def __get_max_new_tokens(self) -> int:
        """
        This method returns the maximum number of new tokens to add to the model for the given task

        :param task: task to get the max new tokens for

        :return maximum number of new tokens
        """
        max_new_tokens = {
            "outcome_type": 1,
            "binary_outcomes": 50,
            "continuous_outcomes": 50,
        }
        return max_new_tokens[self.task]

    def run_task(self) -> Tuple[str, str]:
        '''
        This method runs the task for the given model, task, and split

        :return paths to the output file (json and csv) as a tuple
        '''
        # load dataset prompt templates
        prompts = DatasetTemplates(self.prompt_template)

        # if prompt name is given, apply the prompt template. if not, use all 
        if self.prompt_name is None:
            all_prompt_templates = prompts.all_template_names
            prompt_template_name = all_prompt_templates[0]
        else:
            prompt_template_name = self.prompt_name
        
        prompt = prompts[prompt_template_name]

        # format the dataset with the prompt template
        dataset = [format_example_with_prompt_template(example, prompt) for example in tqdm(self.dataset)]

        # run the task using specified model
        results = []
        pbar = tqdm(dataset)
        for _, example in enumerate(pbar):
            output = self.model.generate_output(example["input"], max_new_tokens=self.max_new_tokens)
            example["output"] = output
            results.append(example)

        # saving results to file
        print(f"Saving outputs for task - {self.task}; prompt - {prompt.get_name()}; model - {self.model_name} to csv and json")
        current_datetime = datetime.now().strftime("%Y%m%d-%H:%M:%S")

        keys_to_drop = [
            "effect_label", 
            "effect_label_code", 
            "is_data_in_figure_graphics", 
            "is_relevant_data_in_table", 
            "is_table_in_graphic_format", 
            "abstract_and_results_xml", 
            "input"
        ]
        # convert into json
        json_file_path = f"{self.output_path}/{self.model_name}_{self.task}_{self.split}_output_{current_datetime}.json"
        save_dataset_to_json(dataset, json_file_path, keys_to_drop)

        # convert into csv
        csv_file_path = f"{self.output_path}/{self.model_name}_{self.task}_{self.split}_output_{current_datetime}.csv"
        save_dataset_to_csv(dataset, csv_file_path, keys_to_drop)

        return json_file_path, csv_file_path

def run_end_to_end_task(model: str, split: str, output_path: str, is_test: bool) -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    '''
    This method runs the end-to-end task for the given model and split
    
    :param model: name of the model to use
    :param split: split of the dataset to run (dev or test)
    :param output_path: path to save the output data
    :param is_test: whether to run the task with only 10 instances for debugging purposes
    
    :return paths to the output files (json and csv) for outcome types and binary and continuous outcomes as a tuple
            output types: (json, csv) and binary outcomes: (json, csv) and continuous outcomes: (json, csv)
    '''
    # First, call the outcome type task
    outcome_type_task_runner = MetaAnalysisTaskRunner(model, "outcome_type", split, output_path, is_test)
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
    binary_outcomes_task_runner = MetaAnalysisTaskRunner(model, "binary_outcomes", split, output_path, is_test, None, outcome_type_json_file_path)
    binary_outcomes_json_file_path, binary_outcomes_csv_file_path = binary_outcomes_task_runner.run_task()

    # Third, call continuous_outcomes task using the output from the outcome_type task
    continuous_outcomes_task_runner = MetaAnalysisTaskRunner(model, "continuous_outcomes", split, output_path, is_test, None, outcome_type_json_file_path)
    continuous_outcomes_json_file_path, continuous_outcomes_csv_file_path = continuous_outcomes_task_runner.run_task()

    return (outcome_type_json_file_path, outcome_type_csv_file_path), (binary_outcomes_json_file_path, binary_outcomes_csv_file_path), (continuous_outcomes_json_file_path, continuous_outcomes_csv_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Clinical Trials Meta Analysis Task")

    parser.add_argument("--model", default="gpt35", choices=["gpt35", "gpt4"], help="what model to run", required=True)
    parser.add_argument("--task", default="outcome_type", choices=['outcome_type', 'binary_outcomes', 'continuous_outcomes', 'end_to_end'], help="type of task to run", required=True)
    parser.add_argument("--split", default="test", choices=["test", "dev"], help="which split of the dataset to run")
    parser.add_argument("--prompt", default=None, help="specific prompt to run. if no specific prompt is given, the first prompt for the given task is run. OPTIONAL")
    parser.add_argument("--output_path", default="./output", help="directory of where the outputs/results should be saved")
    # do --no-test for explicit False
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, help="used for debugging purposes. test will only run random 10 instances from the dataset.")
    
    args = parser.parse_args()

    model = args.model
    task = args.task
    split = args.split
    prompt_name = args.prompt
    output_path = args.output_path
    is_test = args.test

    print("Arguments for the Clinical Trials Meta Analysis Task Runner:")
    print(f"Model:        {model}")
    print(f"Task:         {task}")
    print(f"Split:        {split}")
    print(f"Prompt Name:  {prompt_name}")
    print(f"Output Path:  {output_path}")
    print(f"Is Test:      {is_test}")
    print()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    if task == "end_to_end":
        outcome_type_task_files, binary_outcomes_task_files, continuous_outcomes_task_files = run_end_to_end_task(model, split, output_path, is_test)
        print(f"Outcome Type task outputs saved to {outcome_type_task_files[0]} and {outcome_type_task_files[1]}")
        print(f"Binary Outcomes task outputs saved to {binary_outcomes_task_files[0]} and {binary_outcomes_task_files[1]}")
        print(f"Continuous Outcomes task outputs saved to {continuous_outcomes_task_files[0]} and {continuous_outcomes_task_files[1]}")
    else:
        task_runner = MetaAnalysisTaskRunner(model, task, split, output_path, is_test, prompt_name)
        json_file_path, csv_file_path = task_runner.run_task()
        print(f"Task outputs saved to {json_file_path} and {csv_file_path}")
    
    