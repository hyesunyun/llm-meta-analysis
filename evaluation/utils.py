from templates import Template
from typing import Dict, List
import os
import json

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")

def format_example_with_prompt_template(example: Dict, prompt_template: type[Template]) -> Dict:
    """
    This method formats each dataset example to the give prompt template

    :param example: dataset instance in dictionary format
    :param prompt_template: Template to use to format

    :return dataset instance with correct input format
    """
    formatted_input = prompt_template.apply(example)
    example["input"] = formatted_input
    return example

def load_dataset_from_json(dataset_filename: str) -> List[Dict]:
    """
    This method loads a dataset in json file from the data folder

    :param dataset_filename: name of the dataset to load

    :return dataset as a list of dictionaries
    """
    dataset_path = os.path.join(DATA_FOLDER_PATH, dataset_filename)
    with open(dataset_path, "r") as dataset_file:
        dataset = json.load(dataset_file)
    return dataset

def load_dataset_from_csv(dataset_filename: str) -> List[Dict]:
    """
    This method loads a dataset in csv file from the data folder

    :param dataset_filename: name of the dataset to load

    :return dataset as a list of dictionaries
    """
    dataset_path = os.path.join(DATA_FOLDER_PATH, dataset_filename)
    with open(dataset_path, "r") as dataset_file:
        dataset = dataset_file.readlines()
    return dataset


