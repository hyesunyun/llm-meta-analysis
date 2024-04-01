from typing import Dict, Tuple, List
from sklearn.metrics import f1_score
import numpy as np

class MetricsCalculator:
    def __init__(self, task: str) -> None:
        self.task = task

    def __get_keys_to_compare(self, data: Dict, is_point_estimates: bool = False) -> Tuple[List[str], List[str]]:
        """
        This method gets the keys to calculate the metrics for the given task

        :param data: dictionary with the data to calculate the metrics

        :return tuple of list of keys to calculate the metrics
        """
        search_key = "_output"
        point_estimates_fields = ['log_odds_ratio_output', 'standardized_mean_difference_output']
        if is_point_estimates:
            relevant_output_fields = [key for key in data.keys() if key in point_estimates_fields]
        else:
            relevant_output_fields = [key for key in data.keys() if search_key in key and key not in point_estimates_fields]
        
        relevant_reference_fields = [key.replace(search_key, "") for key in relevant_output_fields]
        return relevant_output_fields, relevant_reference_fields

    def __calculate_accuracy(self, actual: List[str], predicted: List[str], remove_unknowns: bool = False) -> float:
        """
        This method calculates the accuracy metric

        :param actual: list of actual values
        :param predicted: list of predicted values
        :param remove_unknowns: boolean to remove unknowns

        :return accuracy as a float
        """
        num_actual = len(actual)
        correct = 0
        for i in range(num_actual):
            if remove_unknowns and predicted[i] == "x":
                num_actual -= 1
                continue
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(num_actual)

    def __calculate_f_score(self, data: List[Dict]) -> Dict:
        """
        This method calculates the F1 score for the given task

        :param data: list of dictionaries with the data to calculate the F1 score
        :return: dictionary with the F1 score
        """
        relevant_output_field = "outcome_type_output"
        relevant_reference_field = "outcome_type" 
        string_to_int_labels = {"binary": 0, "continuous": 1, "x": 2}
        metrics = {}

        actual = [example[relevant_reference_field] for example in data]
        predicted = [example[relevant_output_field] for example in data]

        # convert from string to integer labels
        actual = [string_to_int_labels[a] for a in actual]
        predicted = [string_to_int_labels[p] for p in predicted]
        
        f1_scores = f1_score(actual, predicted, average=None).tolist()

        metrics[relevant_reference_field] = {
            "f1_score_binary": f1_scores[0],
            "f1_score_continuous": f1_scores[1],
            "f1_score_unknown": f1_scores[2] if len(f1_scores) > 2 else None
        }
        return metrics
    
    def __calculate_mean_absolute_error(self, actual: List[float], predicted: List[float]) -> float:
        """
        This method calculates the mean absolute error

        :param actual: list of actual values
        :param predicted: list of predicted values

        :return: mean absolute error as a float
        """
        total_num = 0
        list_of_abs_diff = []
        for a, p in zip(actual, predicted):
            if None in (a, p):
                continue
            else:
                total_num += 1
                list_of_abs_diff.append(abs(a - p))

        return sum(list_of_abs_diff) / total_num
    
    def __calculate_standard_error_of_mean_absolute_error(self, actual: List[float], predicted: List[float]) -> float:
        """
        This method calculates the standard error of the mean absolute error

        :param actual: list of actual values
        :param predicted: list of predicted values

        :return: standard error of the mean absolute error as a float
        """
        total_num = 0
        list_of_abs_diff = []
        for a, p in zip(actual, predicted):
            if None in (a, p):
                continue
            else:
                total_num += 1
                list_of_abs_diff.append(a - p)
        return np.std(list_of_abs_diff) / np.sqrt(total_num)
    
    def __calculate_95_confidence_interval_of_mean_absolute_error(self, mean_absolute_error: float, standard_error: float) -> Tuple[float, float]:
        """
        This method calculates the 95% confidence interval of the mean absolute error (mean differences of actual vs predicted values)

        :param mean_absolute_error: mean absolute error
        :param standard_error: standard error of the mean absolute error

        :return: tuple with the lower and upper bounds of the confidence interval
        """
        return mean_absolute_error - 1.96 * standard_error, mean_absolute_error + 1.96 * standard_error

    def __calculate_exact_match_accuracy(self, data: List[Dict], remove_unknowns: bool = False) -> Dict:
        """
        This method calculates the exact match accuracy for the given task

        :param data: list of dictionaries with the data to calculate the accuracy
        :param remove_unknowns: boolean to remove unknowns. If True, then unknowns in the model output are considered correct if the reference is not unknown

        :return: dictionary with the metrics
        """
        item = data[0]
        relevant_output_fields, relevant_reference_fields = self.__get_keys_to_compare(item, False)

        metrics = {}
        for output, reference in zip(relevant_output_fields, relevant_reference_fields):
            actual = [example[reference] for example in data]
            predicted = [example[output] for example in data]

            metrics[reference] = self.__calculate_accuracy(actual, predicted, remove_unknowns)

        
        # get the total accuracy
        if self.task == "binary_outcomes":
            num_parts = 4
        elif self.task == "continuous_outcomes":
            num_parts = 6
        else:
            num_parts = 1

        num_total = len(data)
        correct = 0
        for example in data:
            num_parts_correct = 0
            for output, reference in zip(relevant_output_fields, relevant_reference_fields):
                if remove_unknowns and example[output] == "x" and example[reference] != "x":
                    num_parts_correct += 1 # consider it as correct
                if example[output] == example[reference]:
                    num_parts_correct += 1
            if num_parts_correct == num_parts:
                correct += 1
        metrics["total"] = correct / float(num_total)

        return metrics
    
    def __calculate_partial_match_accuracy(self, data: List[Dict], remove_unknowns: bool = False) -> Dict:
        """
        This method calculates the partial match accuracy for the given task

        :param data: list of dictionaries with the data to calculate the accuracy
        :param remove_unknowns: boolean to remove unknowns. If True, then unknowns in the model output are considered correct if the reference is not unknown
        :return: dictionary with metrics
        """
        # different levels of partial matching metrics so this could be 1, 2, 3, 4, 5 etc.
        if self.task == "binary_outcomes":
            num_matches = [1, 2, 3]
        elif self.task == "continuous_outcomes":
            num_matches = [1, 2, 3, 4, 5]
        else:
            num_matches = [1] # this is the default. for outcome type task, this is same as exact match accuracy

        item = data[0]
        relevant_output_fields, relevant_reference_fields = self.__get_keys_to_compare(item, False)

        metrics = {}
        
        for num_match in num_matches:
            num_total = len(data)
            correct = 0
            for example in data:
                num_parts_correct = 0
                for output, reference in zip(relevant_output_fields, relevant_reference_fields):
                    if remove_unknowns and example[output] == "x" and example[reference] != "x":
                        num_parts_correct += 1 # consider it as correct
                    if example[output] == example[reference]:
                        num_parts_correct += 1
                if num_parts_correct >= num_match:
                    correct += 1

            metrics[f"partial_match_accuracy_{num_match}"] = correct / float(num_total)

        return metrics

    def __calculate_number_of_model_unknowns(self, data: List[Dict]) -> Dict:
        """
        This method calculates the number of unknowns or "x" in the data only given by the model

        :param data: list of dictionaries with the data to calculate the number of unknowns
        :return: Dictionary with the number of unknowns for each field and total
        """
        item = data[0]
        relevant_output_fields, relevant_reference_fields = self.__get_keys_to_compare(item, False)
        
        metrics = {}
        for i, field in enumerate(relevant_output_fields):
            unknowns = sum([1 for example in data if (example[field] == "x" and example[relevant_reference_fields[i]] != "x") or example[field] == "unknown"])
            metrics[relevant_reference_fields[i]] = unknowns
        metrics["total"] = sum(metrics.values())
        return metrics

    def __calculate_number_of_reference_unknowns(self, data: List[Dict]) -> Dict:
        """
        This method calculates the number of unknowns or "x" in the data by human annotators

        :param data: list of dictionaries with the data to calculate the number of unknowns
        :return: Dictionary with the number of unknowns for each field and total
        """
        item = data[0]
        _, relevant_reference_fields = self.__get_keys_to_compare(item, False)
        
        metrics = {}
        for i, _ in enumerate(relevant_reference_fields):
            unknowns = sum([1 for example in data if example[relevant_reference_fields[i]] == "x"])
            metrics[relevant_reference_fields[i]] = unknowns
        metrics["total"] = sum(metrics.values())
        return metrics
    
    def __calculate_num_of_chunked_instances(self, data: List[Dict]) -> int:
        """
        This method calculates the number of chunked instances in the data

        :param data: list of dictionaries with the data to calculate the number of chunked instances
        :return: number of chunked instances for each field and total
        """
        num_chunked_instances = sum([1 for example in data if "is_chunked" in example and example["is_chunked"]])
        return num_chunked_instances
    
    def __calculate_point_estimates_metrics(self, data: List[Dict]) -> Dict:
        """
        This method calculates the metrics for the derived point estimates

        :param data: list of dictionaries with the data to calculate the metrics
        :return: dictionary with the metrics (mean absolute error, mean squared error, root mean squared error)
        """
        item = data[0]
        relevant_output_fields, relevant_reference_fields = self.__get_keys_to_compare(item, True)

        metrics = {}
        
        for output, reference in zip(relevant_output_fields, relevant_reference_fields):
            actual = [example[reference] for example in data]
            predicted = [example[output] for example in data]

            # calculate the mean absolute error
            mean_absolute_error = self.__calculate_mean_absolute_error(actual, predicted)
            # calculate the standard error of the mean absolute error
            standard_error = self.__calculate_standard_error_of_mean_absolute_error(actual, predicted)
            # calculate the 95% confidence interval of the mean absolute error
            lower_bound, upper_bound = self.__calculate_95_confidence_interval_of_mean_absolute_error(mean_absolute_error, standard_error)
        
            metrics[reference] = {"mean_absolute_error": mean_absolute_error, "standard_error_of_mae": standard_error, "95_confidence_interval_of_mae": (lower_bound, upper_bound)}

        return metrics

    def calculate_metrics(self, data: List[Dict]) -> Dict:
        """
        This method calculates the metrics for the given task

        :param data: list of dictionaries with the data to calculate the metrics
        :return: dictionary with the metrics
        """
        metrics = {}

        # calculate number of unknowns
        metrics["number_of_model_unknowns"] = self.__calculate_number_of_model_unknowns(data)
        metrics["number_of_reference_unknowns"] = self.__calculate_number_of_reference_unknowns(data)

        # calculate exact match accuracy
        metrics["exact_match_accuracy"] = self.__calculate_exact_match_accuracy(data)
        metrics["exact_match_accuracy_remove_unknowns"] = self.__calculate_exact_match_accuracy(data, True)
        # calcualte partial match accuracy
        metrics["partial_match_accuracy"] = self.__calculate_partial_match_accuracy(data)
        metrics["partial_match_accuracy_remove_unknowns"] = self.__calculate_partial_match_accuracy(data, True)

        if self.task == "outcome_type":
            # calculate the F score for outcome type
            metrics["outcome_type_f_score"] = self.__calculate_f_score(data)

        if self.task == "binary_outcomes" or self.task == "continuous_outcomes":
            # calculate the metrics for point estimates
            metrics["point_estimates"] = self.__calculate_point_estimates_metrics(data)
            metrics["num_of_chunked_instances"] = self.__calculate_num_of_chunked_instances(data)

        return metrics

