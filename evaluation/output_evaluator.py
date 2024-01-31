# postprocess the output of the model and calculate the metrics

import argparse
import os
from utils import (load_json_file)
from metrics_calculator import MetricsCalculator

class MetaAnalysisTaskEvaluator:
    def __init__(self, task: str, output_path: str) -> None:
        self.task = task
        self.output_path = output_path

        self.data = None
        self.metrics_calculator = MetricsCalculator()

        self._load_data()

    def _load_data(self) -> None:
        """
        This method loads the data for the given task using the output path
        """
        self.data = load_json_file(self.output_path)

    def run_evaluation(self) -> None:
        """
        This method runs the evaluation for the given task
        """
        # TODO: Implement run_evaluation
        # figure out what preprocessing to do depending on the task
        # do the preprocessing (parse the output of the model into format that can be used by the metrics calculator)
        # calculate the metrics (accuracy, precision, recall, f1 for both exact match and partial match)
        # the MetricsCalculator class should be able to do this
        pass

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
    