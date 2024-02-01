import argparse
import os
from utils import (
    format_example_with_prompt_template, 
    load_dataset_from_json, 
    get_xml_content_by_pmcid,
    save_dataset_to_json,
    save_dataset_to_csv
)
from typing import Dict, List, Optional
from tqdm import tqdm
from templates import DatasetTemplates
from models import Model, GPT35, GPT4
from datetime import datetime
import random

class MetaAnalysisTaskRunner:
    def __init__(self, model_name: str, task: str, split: str, output_path: str, is_test: bool, prompt_name: Optional[str]=None) -> None:
        self.model_name = model_name
        self.task = task
        self.split = split
        self.prompt_name = prompt_name
        self.output_path = output_path
        self.is_test = is_test

        self.prompt_template = None
        self.dataset = None
        self.model = None

        self._load_prompt_template()
        self._load_dataset()
        self._load_model()

    def _load_prompt_template(self) -> str:
        """
        This method loads the prompt template for the given task

        :return string of the full prompt template
        """
        # this can be deleted if we are not going to use any models other than OpenAI ones
        # if we use some open source models, we should discriminate model too and add model name to path
        self.prompt_template = task
    
    def _load_dataset(self) -> List[Dict]:
        """
        This method loads the dataset (test split)

        :return dataset as a list of dictionaries
        """
        dataset_filename = "meta_analysis_dataset.json"
        dataset = load_dataset_from_json(dataset_filename)

        # get the correct split to run the task
        if self.split == "dev":
            dataset = [example for example in dataset if example["split"] == "DEV"]
        else:
            # filter out dataset to only test (evaluation) split
            dataset = [example for example in dataset if example["split"] == "TEST"]

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

        self.dataset = dataset

    def _load_model(self) -> Model:
        """
        This method loads the model requested for the task

        :return Model object
        """
        model_class_mapping = {"gpt35": GPT35, "gpt4": GPT4}
        model_class = model_class_mapping[self.model_name]
        self.model = model_class()

    def run_task(self):
        # load dataset prompt templates
        prompts = DatasetTemplates(self.templates)

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
            output = self.model.generate_output(example["input"])
            example["output"] = output
            results.append(example)

        # saving results to file
        print(f"Saving outputs for task - {self.task}; prompt - {prompt_name}; model - {self.model_name} to csv and json")
        current_datetime = datetime.now().strftime("%Y%m%d")

        # convert into json
        json_file_path = f"{self.output_path}/{self.model_name}_{self.task}_output_{current_datetime}.json"
        save_dataset_to_json(dataset, json_file_path)

        # convert into csv
        csv_file_path = f"{self.output_path}/{self.model_name}_{self.task}_output_{current_datetime}.csv"
        save_dataset_to_csv(dataset, csv_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Clinical Trials Meta Analysis Task")

    parser.add_argument("--model", default="gpt35", choices=["gpt35", "gpt4"], help="what model to run", required=True)
    parser.add_argument("--task", default="outcome_type", choices=['outcome_type', 'binary_outcomes', 'continuous_outcomes'], help="type of task to run", required=True)
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
    print(f"model: {model}")
    print(f"task: {task}")
    print(f"split: {split}")
    print(f"prompt_name: {prompt_name}")
    print(f"output_path: {output_path}")
    print(f"is_text: {is_test}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("output path did not exist. directory was created.")

    task_runner = MetaAnalysisTaskRunner(model, task, split, output_path, is_test, prompt_name)
    task_runner.run_task()
    