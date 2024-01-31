from templates import Template
from typing import Dict, List
import os
import json
import csv

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")
XML_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data", "abstract_and_results_xml_files")

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

def load_json_file(file_path: str) -> List[Dict]:
    """
    This method loads a json file from the given file path

    :param file_path: name of the file to load

    :return objects as a list of dictionaries
    """
    with open(file_path, "r") as file:
        json_file = json.load(file)
    return json_file

def save_dataset_to_json(dataset: List[Dict], file_path: str) -> None:
    """
    This method saves a dataset in json file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save

    """
    with open(file_path, "w", encoding='utf-8') as file:
            json.dump(dataset, file)

def save_dataset_to_csv(dataset: List[Dict], file_path: str) -> None:
    """
    This method saves a dataset in csv file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save

    """
    keys = dataset[0].keys()
    with open(file_path, "w", newline='', encoding='utf-8') as file:
        dict_writer = csv.DictWriter(file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)

def get_xml_content_by_pmcid(pmcid: str) -> str:
    """
    This method gets the xml file contents of a given pmcid

    :param pmcid: pmcid of the xml file

    :return xml file contents as a string
    """
    xml_filename = f"{pmcid}.xml"
    xml_path = os.path.join(XML_FOLDER_PATH, xml_filename)
    with open(xml_path, "r") as xml_file:
        xml_content = xml_file.read()
    return xml_content

