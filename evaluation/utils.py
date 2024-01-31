from templates import Template
from typing import Dict, List
import os
import json

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

