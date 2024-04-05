import argparse
import os
from utils import (
    load_json_file,
    convert_character_to_string_outcome_type,
    save_json_file,
    calculate_log_odds_ratio,
    calculate_standardized_mean_difference,
    clean_yaml_output,
    parse_multiple_yaml_output,
    aggregate_yaml_output_for_binary_outcomes,
    aggregate_yaml_output_for_continuous_outcomes
)
from calculate_metrics import MetricsCalculator
import yaml
import json

DEFAULT_BINARY_OUTCOMES_DICT = {
    "intervention": {
        "events": "x",
        "group_size": "x"
    },
    "comparator": {
        "events": "x",
        "group_size": "x"
    }
}

DEFAULT_CONTINUOUS_OUTCOMES_DICT = {
    "intervention": {
        "mean": "x",
        "standard_deviation": "x",
        "group_size": "x"
    },
    "comparator": {
        "mean": "x",
        "standard_deviation": "x",
        "group_size": "x"
    }
}

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")
XML_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data", "no_attributes_xml_files")
MD_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data", "no_attributes_markdown_files")

class MetaAnalysisTaskEvaluator:
    def __init__(self, task: str, output_path: str, metrics_path: str, pmc_files_path: str) -> None:
        self.task = task
        self.output_path = output_path
        self.metrics_path = metrics_path
        self.pmc_files_path = pmc_files_path if pmc_files_path is not None else MD_FOLDER_PATH

        self.data = None
        self.metrics_calculator = MetricsCalculator(self.task)

        self.__load_data()

    def __load_data(self) -> None:
        """
        This method loads the data for the given task using the output path
        """
        self.data = load_json_file(self.output_path)

    def __preprocess_outcome_type_results(self) -> None:
        """
        This method preprocesses the data for outcome_type task
        """
        for example in self.data:
            model_output = example["output"]
            output_outcome_type = convert_character_to_string_outcome_type(model_output)
            new_item = {"outcome_type_output": output_outcome_type}
            example.update(new_item)

    def __preprocess_binary_outcomes_results(self) -> None:
        """
        This method preprocesses the data for binary_outcomes task
        """
        for example in self.data:

            # Keep track of the pmcid we are processing
            pmcid = example["pmcid"]

            # need to also fill in the x to make it easier to compare with the output results
            fields_to_check = ["intervention_events", "intervention_group_size", "comparator_events", "comparator_group_size"]
            for field in fields_to_check:
                if example[field] == "" or example[field] is None:
                    example.update({field: "x"}) # all our templates use "x" to represent "not available"

            ie = example["intervention_events"]
            it = example["intervention_group_size"]
            ce = example["comparator_events"]
            ct = example["comparator_group_size"]
            example.update({
                "log_odds_ratio": calculate_log_odds_ratio(ie, ce, it, ct)
            })

            model_output = example["output"]
            # aggregate the output if the input was chunked because the output will have multiple parts
            if "is_chunked" in example and example["is_chunked"]:
                yaml_strings_list = parse_multiple_yaml_output(model_output)
                yaml_dict_list = []
                for yaml_string in yaml_strings_list:
                    try:
                        cleaned_yaml_string = clean_yaml_output(yaml_string)
                        yaml_dict_list.append(yaml.safe_load(cleaned_yaml_string))
                    except:
                        print(f"Error parsing yaml string: {cleaned_yaml_string}")
                        yaml_dict_list.append(DEFAULT_BINARY_OUTCOMES_DICT)

                output_dict = aggregate_yaml_output_for_binary_outcomes(yaml_dict_list, pmcid, self.pmc_files_path)
            else:
                try:
                    cleaned_yaml_string = clean_yaml_output(model_output)
                    output_dict = yaml.safe_load(cleaned_yaml_string)
                except:
                    print(f"Error parsing yaml string: {cleaned_yaml_string}")
                    output_dict = DEFAULT_BINARY_OUTCOMES_DICT

            if "intervention" in output_dict or "comparator" in output_dict:
                intervention = output_dict["intervention"] if "intervention" in output_dict else {}
                comparator = output_dict["comparator"] if "comparator" in output_dict else {}

                ie_output = intervention["events"] if intervention and "events" in intervention else "x"
                it_output = intervention["group_size"] if intervention and "group_size" in intervention else "x"
                ce_output = comparator["events"] if comparator and "events" in comparator else "x"
                ct_output = comparator["group_size"] if comparator and "group_size" in comparator else "x"
                new_item = {
                    "intervention_events_output": ie_output,
                    "intervention_group_size_output": it_output,
                    "comparator_events_output": ce_output,
                    "comparator_group_size_output": ct_output,
                    "log_odds_ratio_output": calculate_log_odds_ratio(ie_output, ce_output, it_output, ct_output)
                }
                example.update(new_item)
            else:
                new_item = {
                    "intervention_events_output": "x",
                    "intervention_group_size_output": "x",
                    "comparator_events_output": "x",
                    "comparator_group_size_output": "x",
                    "log_odds_ratio_output": None
                }
                example.update(new_item)

    def __preprocess_continuous_outcomes_results(self) -> None:
        """
        This method preprocesses the data for continuous_outcomes task
        """
        for example in self.data:

            # Keep track of the pmcid we are processing
            pmcid = example["pmcid"]

            # need to also fill in the x to make it easier to compare with the output results
            fields_to_check = ["intervention_mean", "intervention_standard_deviation", "intervention_group_size", "comparator_mean", "comparator_standard_deviation", "comparator_group_size"]
            for field in fields_to_check:
                if example[field] == "" or example[field] is None:
                    example.update({field: "x"}) # all our templates use "x" to represent "not available"

            im = example["intervention_mean"]
            isd = example["intervention_standard_deviation"]
            cm = example["comparator_mean"]
            csd = example["comparator_standard_deviation"]
            example.update({
                "standardized_mean_difference": calculate_standardized_mean_difference(im, cm, isd, csd),
            })

            model_output = example["output"]
            # aggregate the output if the input was chunked because the output will have multiple parts
            if "is_chunked" in example and example["is_chunked"]:
                yaml_strings_list = parse_multiple_yaml_output(model_output)
                yaml_dict_list = []
                for yaml_string in yaml_strings_list:
                    try:
                        cleaned_yaml_string = clean_yaml_output(yaml_string)
                        yaml_dict_list.append(yaml.safe_load(cleaned_yaml_string))
                    except:
                        print(f"Error parsing yaml string: {cleaned_yaml_string}")
                        yaml_dict_list.append(DEFAULT_CONTINUOUS_OUTCOMES_DICT)
                output_dict = aggregate_yaml_output_for_continuous_outcomes(yaml_dict_list, pmcid, self.pmc_files_path)
            else:
                try:
                    cleaned_yaml_string = clean_yaml_output(model_output)
                    output_dict = yaml.safe_load(cleaned_yaml_string)
                except:
                    print(f"Error parsing yaml string: {cleaned_yaml_string}")
                    output_dict = DEFAULT_CONTINUOUS_OUTCOMES_DICT

            if "intervention" in output_dict or "comparator" in output_dict:
                intervention = output_dict["intervention"] if "intervention" in output_dict else {}
                comparator = output_dict["comparator"] if "comparator" in output_dict else {}

                im_output = intervention["mean"] if intervention and "mean" in intervention else "x"
                isd_output = intervention["standard_deviation"] if intervention and "standard_deviation" in intervention else "x"
                it_output = intervention["group_size"] if intervention and "group_size" in intervention else "x"
                cm_output = comparator["mean"] if comparator and "mean" in comparator else "x"
                csd_output = comparator["standard_deviation"] if comparator and "standard_deviation" in comparator else "x"
                ct_output = comparator["group_size"] if comparator and "group_size" in comparator else "x"
                new_item = {
                    "intervention_mean_output": im_output,
                    "intervention_standard_deviation_output": isd_output,
                    "intervention_group_size_output": it_output,
                    "comparator_mean_output": cm_output,
                    "comparator_standard_deviation_output": csd_output,
                    "comparator_group_size_output": ct_output,
                    "standardized_mean_difference_output": calculate_standardized_mean_difference(im_output, cm_output, isd_output, csd_output)
                }
                example.update(new_item)
            else:
                new_item = {
                    "intervention_mean_output": "x",
                    "intervention_standard_deviation_output": "x",
                    "intervention_group_size_output": "x",
                    "comparator_mean_output": "x",
                    "comparator_standard_deviation_output": "x",
                    "comparator_group_size_output": "x",
                    "standardized_mean_difference_output": None
                }
                example.update(new_item)

    def run_evaluation(self) -> None:
        """
        This method runs the evaluation for the given task
        """
        
        if self.task == "outcome_type":
            self.__preprocess_outcome_type_results()
        elif self.task == "binary_outcomes":
            self.__preprocess_binary_outcomes_results()
        elif self.task == "continuous_outcomes":
            self.__preprocess_continuous_outcomes_results()

        # calculate the metrics
        # print("Pre-processed DATA:")
        # print(json.dumps(self.data, indent=4))
        metrics = self.metrics_calculator.calculate_metrics(self.data)
        
        # print and save the metrics to a file
        print("Metrics for the task:")
        print(json.dumps(metrics, indent=4))

        output_file_name = self.output_path.split("/")[-1]
        save_json_file(f"{self.metrics_path}/{output_file_name}_metrics.json", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating Outputs for Clinical Trials Meta Analysis Task")

    parser.add_argument("--task", default="outcome_type", choices=['outcome_type', 'binary_outcomes', 'continuous_outcomes'], help="type of task to run", required=True)
    parser.add_argument("--output_path", default="./output", help="directory of where the outputs/results are saved")
    parser.add_argument("--metrics_path", default="./metrics", help="directory of where the metrics should be saved")
    parser.add_argument("--pmc_files_path", default=None, help="directory of where the PMC files are saved")
    
    args = parser.parse_args()

    task = args.task
    output_path = args.output_path
    metrics_path = args.metrics_path
    pmc_files_path = args.pmc_files_path

    print("Arguments Provided for the Clinical Trials Meta Analysis Task Evaluator:")
    print(f"Task:         {task}")
    print(f"Output Path:  {output_path}")
    print(f"Metrics Path: {metrics_path}")
    print(f"PMC Files Path: {pmc_files_path}")
    print()

    if not os.path.exists(output_path):
        print("ERROR: output path does not exist.")
        exit(1)

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
        print("Metrics path did not exist. Directory was created.")

    if not os.path.exists(pmc_files_path):
        print("ERROR: PMC files path does not exist.")

    task_evaluator = MetaAnalysisTaskEvaluator(task, output_path, metrics_path)
    task_evaluator.run_evaluation()
    