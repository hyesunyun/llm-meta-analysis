import argparse
import os
from utils import (
    load_json_file,
    convert_character_to_string_outcome_type,
    save_json_file
)
from metrics_calculator import MetricsCalculator
import yaml
import datetime

class MetaAnalysisTaskEvaluator:
    def __init__(self, task: str, output_path: str) -> None:
        self.task = task
        self.output_path = output_path

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
            model_output = example["output"]

            # need to also fill in the x to make it easier to compare with the output results
            fields_to_check = ["intervention_events", "intervention_group_size", "comparator_events", "comparator_group_size"]
            for field in fields_to_check:
                if example[field] == "" or example[field] is None:
                    example.update({field: "x"}) # all our templates use "x" to represent "not available"

            output_dict = yaml.load(model_output)
            new_item = {
                "intervention_events_output": output_dict["intervention"]["events"],
                "intervention_group_size_output": output_dict["intervention"]["group_size"],
                "comparator_events_output": output_dict["comparator"]["events"],
                "comparator_group_size_output": output_dict["comparator"]["group_size"],
            }
            example.update(new_item)

    def __preprocess_continuous_outcomes_results(self) -> None:
        """
        This method preprocesses the data for continuous_outcomes task
        """
        for example in self.data:
            model_output = example["output"]

            # need to also fill in the x to make it easier to compare with the output results
            fields_to_check = ["intervention_mean", "intervention_standard_deviation", "intervention_group_size", "comparator_mean", "comparator_standard_deviation", "comparator_group_size"]
            for field in fields_to_check:
                if example[field] == "" or example[field] is None:
                    example.update({field: "x"}) # all our templates use "x" to represent "not available"

            output_dict = yaml.load(model_output)
            new_item = {
                "intervention_mean_output": output_dict["intervention"]["mean"],
                "intervention_standard_deviation_output": output_dict["intervention"]["standard_deviation"],
                "intervention_group_size_output": output_dict["intervention"]["group_size"],
                "comparator_mean_output": output_dict["comparator"]["mean"],
                "comparator_standard_deviation_output": output_dict["comparator"]["standard_deviation"],
                "comparator_group_size_output": output_dict["comparator"]["group_size"],
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
        metrics = self.metrics_calculator.calculate_metrics(self.data)
        
        # print and save the metrics to a file
        print("Metrics for the task:")
        print(metrics)

        current_datetime = datetime.now().strftime("%Y%m%d")
        save_json_file(f"{self.output_path}/{self.task}_metrics_{current_datetime}.json", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating Outputs for Clinical Trials Meta Analysis Task")

    parser.add_argument("--task", default="outcome_type", choices=['outcome_type', 'binary_outcomes', 'continuous_outcomes'], help="type of task to run", required=True)
    parser.add_argument("--output_path", default="./output", help="directory of where the outputs/results should be saved")
    
    args = parser.parse_args()

    task = args.task
    output_path = args.output_path

    print("Arguments for the Clinical Trials Meta Analysis Task Evaluator:")
    print(f"task: {task}")
    print(f"output_path: {output_path}")

    if not os.path.exists(output_path):
        print("ERROR: output path did not exist.")
        exit(1)

    task_evaluator = MetaAnalysisTaskEvaluator(task, output_path)
    task_evaluator.run_evaluation()
    