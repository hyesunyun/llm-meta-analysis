from templates import Template
from typing import Dict, List, Optional
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

def save_json_file(file_path: str, data: Dict) -> None:
    """
    This method saves a dictionary to a json file

    :param file_path: name of the file to save
    :param data: data to save
    """
    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(data, file)

def save_dataset_to_json(dataset: List[Dict], file_path: str, columns_to_drop: Optional[List[str]] = None) -> None:
    """
    This method saves a dataset in json file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save

    """
    if columns_to_drop is not None:
        dataset = [{k: v for k, v in d.items() if k not in columns_to_drop} for d in dataset]
    with open(file_path, "w", encoding='utf-8') as file:
            json.dump(dataset, file)

def save_dataset_to_csv(dataset: List[Dict], file_path: str, columns_to_drop: Optional[List[str]] = None) -> None:
    """
    This method saves a dataset in csv file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save

    """
    if columns_to_drop is not None:
        dataset = [{k: v for k, v in d.items() if k not in columns_to_drop} for d in dataset]
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

def convert_character_to_string_outcome_type(outcome_type: str) -> str:
    """
    This method converts the outcome type from character to string

    :param outcome_type: outcome type as character

    :return outcome type as string
    """
    character_to_string_mapping = {"b": "binary", "c": "continuous", "x": "unknown"}
    return character_to_string_mapping[outcome_type]

def convert_string_to_character_outcome_type(outcome_type: str) -> str:
    """
    This method converts the outcome type from string to character

    :param outcome_type: outcome type as string

    :return outcome type as character
    """
    string_to_character_mapping = {"binary": "b", "continuous": "c", "unknown": "x"}
    return string_to_character_mapping[outcome_type]

def calculate_odds_ratio(intervention_events: int, control_events: int, intervention_total: int, control_total: int) -> float:
    """
    This method calculates the odds ratio given the values

    :param intervention_events: value of intervention_events
    :param control_events: value of control_events
    :param intervention_total: value of intervention_total
    :param control_total: value of control_total

    :return odds ratio
    """
    intervention_nonevents = intervention_total - intervention_events
    control_nonevents = control_total - control_events

    intervention_events, control_events, intervention_nonevents, control_nonevents = check_and_apply_zero_correction(intervention_events, control_events, intervention_nonevents, control_nonevents)

    return (intervention_events * control_nonevents) / (control_events * intervention_nonevents)

def calculate_standard_error_log_odds_ratio(intervention_events: int, control_events: int, intervention_total: int, control_total: int) -> float:
    """
    This method calculates the standard error of the log odds ratio given the values

    :param intervention_events: value of intervention_events
    :param control_events: value of control_events
    :param intervention_total: value of intervention_total
    :param control_total: value of control_total

    :return standard error of the log odds ratio
    """
    intervention_nonevents = intervention_total - intervention_events
    control_nonevents = control_total - control_events

    intervention_events, control_events, intervention_nonevents, control_nonevents = check_and_apply_zero_correction(intervention_events, control_events, intervention_nonevents, control_nonevents)

    return ((1 / intervention_events) + (1 / intervention_nonevents) + (1 / control_events) + (1 / control_nonevents)) ** 0.5

def calculate_risk_ratio(intervention_events: int, control_events: int, intervention_total: int, control_total: int) -> float:
    """
    This method calculates the risk ratio given the values

    :param intervention_events: value of intervention_events
    :param control_events: value of control_events
    :param intervention_total: value of intervention_total
    :param control_total: value of control_total

    :return risk ratio
    """
    intervention_nonevents = intervention_total - intervention_events
    control_nonevents = control_total - control_events

    intervention_events, control_events, intervention_nonevents, control_nonevents = check_and_apply_zero_correction(intervention_events, control_events, intervention_nonevents, control_nonevents)

    return (intervention_events / (intervention_events + intervention_nonevents)) / (control_events / (control_events + control_nonevents))

def calculate_standard_error_log_risk_ratio(intervention_events: int, control_events: int, intervention_total: int, control_total: int) -> float:
    """
    This method calculates the standard error of the log risk ratio given the values

    :param intervention_events: value of intervention_events
    :param control_events: value of control_events
    :param intervention_total: value of intervention_total
    :param control_total: value of control_total

    :return standard error of the log risk ratio
    """
    intervention_nonevents = intervention_total - intervention_events
    control_nonevents = control_total - control_events

    intervention_events, control_events, intervention_nonevents, control_nonevents = check_and_apply_zero_correction(intervention_events, control_events, intervention_nonevents, control_nonevents)

    return ((1 / intervention_events) + (1 / control_events) - (1 / (intervention_events + intervention_nonevents)) - (1 / (control_events + control_nonevents))) ** 0.5

def check_and_apply_zero_correction(intervention_events: int, control_events: int, intervention_nonevents: int, control_nonevents: int) -> float:
    """
    This method applies a zero correction to a contingency table if needed

    :param intervention_events: intervention_events
    :param control_events: control_events
    :param intervention_nonevents: intervention_nonevents
    :param control_nonevents: control_nonevents

    :return all values with zero correction (if applied)
    """
    # Haldane-Anscombe correction (algorithm used by Review Manager - RevMan software for meta-analysis)
    # This involves adding 0.5 to each cell value if any of the cells in the contingency table contain a zero
    # Except when intervention_events and control_events = 0 or intervention_nonevents and control_nonevents = 0, OR and RR is undefined
    if (intervention_events == 0) or (control_events == 0) or (intervention_nonevents == 0) or (control_nonevents == 0):
        if (intervention_events == 0 and control_events == 0) or (intervention_nonevents == 0 and control_nonevents == 0):
            print("Error: Undefined results.")
            return None, None, None, None
        else:
            intervention_events += 0.5
            control_events += 0.5
            intervention_nonevents += 0.5
            control_nonevents += 0.5

    return intervention_events, control_events, intervention_nonevents, control_nonevents

def calculate_mean_difference(intervention_mean: float, control_mean: float) -> float:
    """
    This method calculates the mean difference given the values

    :param intervention_mean: value of intervention_mean
    :param control_mean: value of control_mean

    :return mean difference
    """
    return intervention_mean - control_mean

def calculate_standard_error_mean_difference(intervention_sd: float, control_sd: float, intervention_total: int, control_total: int) -> float:
    """
    This method calculates the standard error of the mean difference given the values

    :param intervention_sd: value of intervention_sd
    :param control_sd: value of control_sd
    :param intervention_total: value of intervention_total
    :param control_total: value of control_total

    :return standard error of the mean difference
    """
    return ((intervention_sd ** 2 / intervention_total) + (control_sd ** 2 / control_total)) ** 0.5