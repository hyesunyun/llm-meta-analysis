import argparse
import os
from utils import format_example_with_prompt_template, load_dataset_from_json

from typing import Dict, List, Optional

from tqdm import tqdm
import os

class MetaAnalysisTaskRunner:
    def __init__(self, model: str, task: str, output_path: str, is_test: bool, prompt_name: Optional[str]=None):
        self.model = model
        self.task = task
        self.prompt_name = prompt_name
        self.output_path = output_path
        self.is_test = is_test

        self.prompt_template = None
        self.dataset = None

        self._load_prompt_template()
        self._load_dataset()

    def _load_prompt_template(self) -> str:
        """
        This method loads the prompt template for the given task

        :return string of the full prompt template
        """
        # this can be deleted if we are not going to use any models other than OpenAI ones
        # if we use some open source models, we should discriminate model too and add model name to path
        return task
    
    def _load_dataset(self) -> List[Dict]:

    def run_task(self):
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Clinical Trials Meta Analysis Task")

    parser.add_argument("--model", default="gpt35", choices=["gpt35", "gpt4"], help="what model to run", required=True)
    parser.add_argument("--task", default="outcome-type", choices=['outcome-type', 'binary-outcome-table', 'continuous-outcome-table'], help="type of task to run", required=True)
    parser.add_argument("--prompt", default=None, help="specific prompt to run. if no specific prompt is given, all prompts related to given task are run. OPTIONAL")
    parser.add_argument("--output_path", default="./output", help="directory of where the outputs/results should be saved")
    # do --no-test for explicit False
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, help="whether this is a test run or not. test will only run 10 instances from the dataset.")
    
    args = parser.parse_args()

    model = args.model
    task = args.task
    prompt_name = args.prompt
    output_path = args.output_path
    is_test = args.test

    print("Arguments for the Clinical Trials Meta Analysis Task Runner:")
    print(f"model: {model}")
    print(f"task: {task}")
    print(f"prompt_name: {prompt_name}")
    print(f"output_path: {output_path}")
    print(f"is_text: {is_test}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("output path did not exist. directory was created.")

    task_runner = MetaAnalysisTaskRunner(model, task, output_path, is_test, prompt_name)
    task_runner.run_task()
    