from templates import Template
from typing import Dict, List, Optional, Any, Tuple
import os
import json
import csv
import re
from statistics import mode
from collections import Counter
from statsmodels.stats.meta_analysis import (
    effectsize_smd,
    effectsize_2proportions
)
import numpy as np

def format_example_with_prompt_template(example: Dict, prompt_template: Template) -> Dict:
    """
    This method formats each dataset example to the give prompt template

    :param example: dataset instance in dictionary format
    :param prompt_template: Template to use to format

    :return dataset instance with correct input format
    """
    formatted_input = prompt_template.apply(example)
    example["input"] = formatted_input
    return example


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
        json.dump(data, file, indent=4)


def save_dataset_to_json(dataset: List[Dict], file_path: str, columns_to_drop: Optional[List[str]] = None) -> None:
    """
    This method saves a dataset (dictionary) in json file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save

    """
    if columns_to_drop is not None:
        dataset = [{k: v for k, v in d.items() if k not in columns_to_drop} for d in dataset]
    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(dataset, file, indent=4)


def save_dataset_to_csv(dataset: List[Dict], file_path: str, columns_to_drop: Optional[List[str]] = None) -> None:
    """
    This method saves a dataset (dictionary) in csv file to the data folder

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


def get_xml_content_by_pmcid(pmc_file_path: str, pmcid: str) -> str:
    """
    This method gets the xml file contents of a given pmcid in a given file path
    
    :param pmc_file_path: path to the pmc files
    :param pmcid: pmcid of the xml file

    :return xml file contents as a string
    """
    xml_filename = f"PMC{pmcid}.xml"
    xml_path = os.path.join(pmc_file_path, xml_filename)
    with open(xml_path, "r") as xml_file:
        xml_content = xml_file.read()
    return xml_content


def get_md_content_by_pmcid(pmc_file_path: str, pmcid: str) -> str:
    """
    This method gets the markdown (md) file contents of a given pmcid in a given file path

    :param pmc_file_path: path to the pmc files
    :param pmcid: pmcid of the md file

    :return md file contents as a string
    """
    md_filename = f"PMC{pmcid}.md"
    md_path = os.path.join(pmc_file_path, md_filename)
    with open(md_path, "r") as md_file:
        md_content = md_file.read()
    return md_content


def convert_character_to_string_outcome_type(outcome_type: str) -> str:
    """
    This method converts the outcome type from character to string

    :param outcome_type: outcome type as character

    :return outcome type as string
    """
    character_to_string_mapping = {"A": "binary", "B": "continuous", "C": "x"}  # x is used to represent unknown
    outcome_type = outcome_type.replace("The answer is ", "").replace(".", "").replace("(", "").replace(")",
                                                                                                        "")  # remove any parens, periods, and other known common, extra texts
    # remove any unnecessary text output by finding the first non-space character
    for char in outcome_type:
        if not char.isspace():
            outcome_type = char
            break
    try:
        string_outcome = character_to_string_mapping[outcome_type]
    except:
        string_outcome = "x"  # x is used to represent unknown
    return string_outcome


def convert_string_to_character_outcome_type(outcome_type: str) -> str:
    """
    This method converts the outcome type from string to character

    :param outcome_type: outcome type as string

    :return outcome type as character
    """
    string_to_character_mapping = {"binary": "A", "continuous": "B", "x": "C"}
    return string_to_character_mapping[outcome_type]


def clean_via_regex(string: str):
    """
    Cleans the yaml string using regex
    we look for a match for intervention with events and group size nested fields in that order for binary. 
    we look for a match for intervention with mean, standard deviation, and group size nested fields in that order for continuous. 
    this is the same for comparator field as well.
    if the order of nested fields is wrong, no matches found.
    extra fields will also result in no matches.
    
    :param string: string to clean
    
    :return cleaned string
    """
    regex = r"^intervention:\n+\s+events:\s*(\d+(\.\d+)?|x|unknown|NUMBER).*\n\s+group_size:\s*(\d+(\.\d+)?|x|unknown|NUMBER).*|\n*\s*comparator:\n+\s+events:\s*(\d+(\.\d+)?|x|unknown|NUMBER).*\n\s+group_size:\s*(\d+(\.\d+)?|x|unknown|NUMBER).*|^intervention:\n+\s+mean:\s*\s+(\d+(\.\d+)?|x|unknown|NUMBER).*\n+\s+standard_deviation:\s*\s+(\d+(\.\d+)?|x|unknown|NUMBER).*\n+\s+group_size:\s*\s+(\d+(\.\d+)?|x|unknown|NUMBER).*|\n*\s*comparator:\n+\s+mean:\s*\s+(\d+(\.\d+)?|x|unknown|NUMBER).*\n+\s+standard_deviation:\s*\s+(\d+(\.\d+)?|x|unknown|NUMBER).*\n+\s+group_size:\s*\s+(\d+(\.\d+)?|x|unknown|NUMBER).*"
    matches = re.finditer(regex, string, re.MULTILINE)
    parsed_string = ""
    for _, match in enumerate(matches, start=1):
        parsed_string += match.group()

    return parsed_string


def clean_yaml_output(output: str) -> str:
    """
    This method cleans the yaml output string

    :param output: yaml output

    :return cleaned yaml output
    """
    cleaned_output = output.lower()
    cleaned_output = cleaned_output.replace("```", "").replace("yaml", "").replace("\t", "")
    cleaned_output = cleaned_output.replace("-x\n", "x\n").replace("-\n", "x\n")  # some post processing
    cleaned_output = cleaned_output.replace("NUMBER", "x").replace("N\A", "x")
    cleaned_output = cleaned_output.replace("\n\ncomparator", "\ncomparator")
    cleaned_output = cleaned_output.replace("*", "")

    # use regex for clean up
    cleaned_output = clean_via_regex(cleaned_output)

    return cleaned_output


def parse_multiple_yaml_output(output: str) -> list[str]:
    """
    This method parses the multiple yaml output to list of str yaml

    :param output: yaml output

    :return list of str yaml
    """
    yaml_strings = output.split("\n---\n")
    yaml_strings.pop()
    return yaml_strings


def has_mode(lst: list[Any]) -> bool:
    """
    This method checks if a list has a mode
    """
    count_dict = Counter(lst)
    most_common = count_dict.most_common(1)
    # if the most common value is 1, then there is no mode
    if most_common[0][1] == 1:
        return False
    return True


def get_file_extension(pmcid, directory_path):
    for filename in os.listdir(directory_path):
        if filename.startswith(f"PMC{pmcid}."):
            file_extension = os.path.splitext(filename)[1]
            return file_extension
    return None


def aggregate_yaml_output_for_binary_outcomes(yaml_dict_list: list[Dict], pmcid: str, pmc_files_path: str) -> Dict:
    """
    This method aggregates the yaml outputs for binary outcomes

    :param yaml_json_list: list of yaml in dict
    :param pmcid: pmcid of the article
    :param pmc_files_path: path to the pmc files

    :return aggregated yaml output
    """

    # get the file extension of the given PMC file in the pmc_files_path and get the content of current pmc id
    file_extension = get_file_extension(pmcid, pmc_files_path)
    if file_extension == ".xml":
        file_content = get_xml_content_by_pmcid(pmc_files_path, pmcid)
    elif file_extension == ".md":
        file_content = get_md_content_by_pmcid(pmc_files_path, pmcid)
    else:
        raise ValueError(f"Error: file extension {file_extension} not supported")


    aggregated_output = {"intervention_events": [], "intervention_group_size": [], "comparator_events": [],
                         "comparator_group_size": []}
    for yaml_dict in yaml_dict_list:
        for key in yaml_dict.keys():
            if key not in ["intervention", "comparator"]:
                print(f"Error: key {key} not found in yaml_dict")
                return None
            if key == "intervention":
                if yaml_dict[key] is not None and "events" in yaml_dict[key].keys():
                    aggregated_output["intervention_events"].append(yaml_dict[key]["events"])
                if yaml_dict[key] is not None and "group_size" in yaml_dict[key].keys():
                    aggregated_output["intervention_group_size"].append(yaml_dict[key]["group_size"])
            if key == "comparator":
                if yaml_dict[key] is not None and "events" in yaml_dict[key].keys():
                    aggregated_output["comparator_events"].append(yaml_dict[key]["events"])
                if yaml_dict[key] is not None and "group_size" in yaml_dict[key].keys():
                    aggregated_output["comparator_group_size"].append(yaml_dict[key]["group_size"])

    # first get all the values for a key that are not x and exist in the file content
    # if there is one numeric output in the list, then we set the aggregated value to that numeric value
    # if there is conflicting numeric outputs in the list, we check if there is a mode, if so, we set the aggregated value to the mode
    # if there is no numeric outputs in the list or if there is no mode, then we set the aggregated value to 'x'
    for key in aggregated_output.keys():
        numeric_values = [value for value in aggregated_output[key] if value != "x" and str(value) in file_content]
        if len(numeric_values) == 1:
            aggregated_output[key] = numeric_values[0]
        elif len(numeric_values) > 1 and has_mode(numeric_values):  # get the mode
            aggregated_output[key] = mode(numeric_values)
        else:
            aggregated_output[key] = "x"

    final_yaml_output = {
        "intervention": {
            "events": aggregated_output["intervention_events"],
            "group_size": aggregated_output["intervention_group_size"]
        },
        "comparator": {
            "events": aggregated_output["comparator_events"],
            "group_size": aggregated_output["comparator_group_size"]
        }
    }
    return final_yaml_output


def aggregate_yaml_output_for_continuous_outcomes(yaml_dict_list: list[Dict], pmcid: str, pmc_files_path: str) -> Dict:
    """
    This method aggregates the yaml outputs for continuous outcomes

    :param yaml_json_list: list of yaml in dict
    :param pmcid: pmcid of the article
    :param pmc_files_path: path to the pmc files

    :return aggregated yaml output
    """

    # get the file extension of the given PMC file in the pmc_files_path and get the content of current pmc id
    file_extension = get_file_extension(pmcid, pmc_files_path)
    if file_extension == ".xml":
        file_content = get_xml_content_by_pmcid(pmc_files_path, pmcid)
    elif file_extension == ".md":
        file_content = get_md_content_by_pmcid(pmc_files_path, pmcid)
    else:
        raise ValueError(f"Error: file extension {file_extension} not supported")

    aggregated_output = {"intervention_mean": [], "intervention_standard_deviation": [], "intervention_group_size": [],
                         "comparator_mean": [], "comparator_standard_deviation": [], "comparator_group_size": []}
    for yaml_dict in yaml_dict_list:
        for key in yaml_dict.keys():
            if key not in ["intervention", "comparator"]:
                print(f"Error: key {key} not found in yaml_dict")
                return None
            if key == "intervention":
                if yaml_dict[key] is not None and "mean" in yaml_dict[key].keys():
                    aggregated_output["intervention_mean"].append(yaml_dict[key]["mean"])
                if yaml_dict[key] is not None and "standard_deviation" in yaml_dict[key].keys():
                    aggregated_output["intervention_standard_deviation"].append(yaml_dict[key]["standard_deviation"])
                if yaml_dict[key] is not None and "group_size" in yaml_dict[key].keys():
                    aggregated_output["intervention_group_size"].append(yaml_dict[key]["group_size"])
            if key == "comparator":
                if yaml_dict[key] is not None and "mean" in yaml_dict[key].keys():
                    aggregated_output["comparator_mean"].append(yaml_dict[key]["mean"])
                if yaml_dict[key] is not None and "standard_deviation" in yaml_dict[key].keys():
                    aggregated_output["comparator_standard_deviation"].append(yaml_dict[key]["standard_deviation"])
                if yaml_dict[key] is not None and "group_size" in yaml_dict[key].keys():
                    aggregated_output["comparator_group_size"].append(yaml_dict[key]["group_size"])

    # first get all the values for a key that are not x and exist in the file content
    # if there is one numeric output in the list, then we set the aggregated value to that numeric value
    # if there is conflicting numeric outputs in the list, we check if there is a mode, if so, we set the aggregated value to the mode
    # if there is no numeric outputs in the list or if there is no mode, then we set the aggregated value to 'x'
    for key in aggregated_output.keys():
        numeric_values = [value for value in aggregated_output[key] if value != "x" and str(value) in file_content]
        if len(numeric_values) == 1:
            aggregated_output[key] = numeric_values[0]
        elif len(numeric_values) > 1 and has_mode(numeric_values):  # get the mode
            aggregated_output[key] = mode(numeric_values)
        else:
            aggregated_output[key] = "x"

    final_yaml_output = {
        "intervention": {
            "mean": aggregated_output["intervention_mean"],
            "standard_deviation": aggregated_output["intervention_standard_deviation"],
            "group_size": aggregated_output["intervention_group_size"]
        },
        "comparator": {
            "mean": aggregated_output["comparator_mean"],
            "standard_deviation": aggregated_output["comparator_standard_deviation"],
            "group_size": aggregated_output["comparator_group_size"]
        }
    }
    return final_yaml_output

def convert_string_to_int(value: str) -> int:
    """
    This method converts a string to an int

    :param value: value to convert

    :return int value
    """
    try:
        return int(value.replace(',', ''))
    except:
        return None

def convert_string_to_float(value: str) -> float:
    """
    This method converts a string to a float

    :param value: value to convert

    :return float value
    """
    try:
        return float(value.replace(',', ''))
    except:
        return None

def calculate_log_odds_ratio(intervention_events: int, control_events: int, intervention_total: int,
                             control_total: int) -> Tuple[float, float]:
    """
    This method calculates the log odds ratio given the values.

    :param intervention_events: value of intervention_events
    :param control_events: value of control_events
    :param intervention_total: value of intervention_total
    :param control_total: value of control_total

    :return log odds ratio
    """
    # need to check for x or unknown in the values
    if "x" in (intervention_events, control_events, intervention_total, control_total):
        return (None, None)
    
    # convert any string values to int
    if isinstance(intervention_events, str):
        intervention_events = convert_string_to_int(intervention_events)
    if isinstance(control_events, str):
        control_events = convert_string_to_int(control_events)
    if isinstance(intervention_total, str):
        intervention_total = convert_string_to_int(intervention_total)
    if isinstance(control_total, str):
        control_total = convert_string_to_int(control_total)

    if None in (intervention_events, control_events, intervention_total, control_total):
        return (None, None)
    # check to make sure that events do not exceed total
    if (intervention_events > intervention_total) or (control_events > control_total):
        return (None, None)
    
    try:    
        # Haldane-Anscombe correction (algorithm used by Review Manager - RevMan software for meta-analysis) This
        # involves adding 0.5 to each cell value if any of the cells in the contingency table contain a zero Except when
        # intervention_events and control_events = 0 or intervention_nonevents and control_nonevents = 0, OR is undefined.
        lor, var_eff = effectsize_2proportions(np.array([intervention_events]), 
                                            np.array([intervention_total]),
                                            np.array([control_events]),
                                            np.array([control_total]),
                                            statistic='odds-ratio', zero_correction=0.5)
        return (lor[0], var_eff[0])
    except:
        print(
            f"An exception occurred for calculate log odds ratio - intervention_events: {intervention_events}, "
            f"control_events: {control_events}, intervention_total: {intervention_total}, control_total: "
            f"{control_total}")
        return (None, None)

def calculate_standardized_mean_difference(intervention_mean: float, control_mean: float, intervention_sd: float,
                                           control_sd: float, intervention_group_size: float, control_group_size: float) -> Tuple[float, float]:
    """
    This method calculates the standardized mean difference given the values

    :param intervention_mean: value of intervention_mean
    :param control_mean: value of control_mean
    :param intervention_sd: value of intervention_sd
    :param control_sd: value of control_sd
    :param intervention_group_size: value of intervention_group_size
    :param control_group_size: value of control_group_size

    :return standardized mean difference
    """
    # need to check for x or unknown in the values
    if "x" in (intervention_mean, control_mean, intervention_sd, control_sd, intervention_group_size, control_group_size):
        return (None, None)
    
    # convert any string values to float
    if isinstance(intervention_mean, str):
        intervention_mean = convert_string_to_float(intervention_mean)
    if isinstance(control_mean, str):
        control_mean = convert_string_to_float(control_mean)
    if isinstance(intervention_sd, str):
        intervention_sd = convert_string_to_float(intervention_sd)
    if isinstance(control_sd, str):
        control_sd = convert_string_to_float(control_sd)
    if isinstance(intervention_group_size, str):
        intervention_group_size = convert_string_to_float(intervention_group_size)
    if isinstance(control_group_size, str):
        control_group_size = convert_string_to_float(control_group_size)

    if None in (intervention_mean, control_mean, intervention_sd, control_sd, intervention_group_size, control_group_size):
        return (None, None)

    try: 
        smd, var_eff = effectsize_smd(np.array([intervention_mean]), 
                                      np.array([intervention_sd]),
                                      np.array([intervention_group_size]),
                                      np.array([control_mean]), 
                                      np.array([control_sd]),  
                                      np.array([control_group_size]))
        return (smd[0], var_eff[0])
    except:
        print(
            f"An exception occurred for calculate standardized mean difference - intervention_mean: {intervention_mean}, control_mean: {control_mean}, intervention_sd: {intervention_sd}, control_sd: {control_sd}, intervention_group_size: {intervention_group_size}, control_group_size: {control_group_size}")
        return (None, None)
