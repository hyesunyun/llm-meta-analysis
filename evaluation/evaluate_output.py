import argparse
import os
from utils import (
    load_json_file,
    convert_character_to_string_outcome_type,
    save_json_file,
    calculate_odds_ratio,
    calculate_standard_error_log_odds_ratio,
    calculate_risk_ratio,
    calculate_standard_error_log_risk_ratio,
    calculate_mean_difference,
    calculate_standard_error_mean_difference
)
from evaluation.calculate_metrics import MetricsCalculator
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

            ie = example["intervention_events"]
            it = example["intervention_group_size"]
            ce = example["comparator_events"]
            ct = example["comparator_group_size"]
            example.update({
                "odds_ratio": calculate_odds_ratio(ie, it, ce, ct),
                "se_log_odds_ratio": calculate_standard_error_log_odds_ratio(ie, it, ce, ct),
                "risk_ratio": calculate_risk_ratio(ie, it, ce, ct),
                "se_log_risk_ratio": calculate_standard_error_log_risk_ratio(ie, it, ce, ct),
            })

            output_dict = yaml.load(model_output)
            ie_output = output_dict["intervention"]["events"]
            it_output = output_dict["intervention"]["group_size"]  
            ce_output = output_dict["comparator"]["events"]
            ct_output = output_dict["comparator"]["group_size"]
            new_item = {
                "intervention_events_output": ie_output,
                "intervention_group_size_output": it_output,
                "comparator_events_output": ce_output,
                "comparator_group_size_output": ct_output,
                "odds_ratio_output": calculate_odds_ratio(ie_output, it_output, ce_output, ct_output),
                "se_log_odds_ratio_output": calculate_standard_error_log_odds_ratio(ie_output, it_output, ce_output, ct_output),
                "risk_ratio_output": calculate_risk_ratio(ie_output, it_output, ce_output, ct_output),
                "se_log_risk_ratio_output": calculate_standard_error_log_risk_ratio(ie_output, it_output, ce_output, ct_output),
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

            im = example["intervention_mean"]
            isd = example["intervention_standard_deviation"]
            it = example["intervention_group_size"]
            cm = example["comparator_mean"]
            csd = example["comparator_standard_deviation"]
            ct = example["comparator_group_size"]
            example.update({
                "mean_difference": calculate_odds_ratio(im, cm),
                "se_mean_difference": calculate_standard_error_log_odds_ratio(isd, csd, it, ct),
            })

            im_output = output_dict["intervention"]["mean"]
            isd_output = output_dict["intervention"]["standard_deviation"]
            it_output = output_dict["intervention"]["group_size"]
            cm_output = output_dict["comparator"]["mean"]
            csd_output = output_dict["comparator"]["standard_deviation"]
            ct_output = output_dict["comparator"]["group_size"]
            output_dict = yaml.load(model_output)
            new_item = {
                "intervention_mean_output": im_output,
                "intervention_standard_deviation_output": isd_output,
                "intervention_group_size_output": it_output,
                "comparator_mean_output": cm_output,
                "comparator_standard_deviation_output": csd_output,
                "comparator_group_size_output": ct_output,
                "mean_difference_output": calculate_mean_difference(im_output, cm_output),
                "se_mean_difference_output": calculate_standard_error_mean_difference(im_output, isd_output, it_output, cm_output, csd_output, ct_output),
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
    