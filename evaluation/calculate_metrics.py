from typing import Dict, List

class MetricsCalculator:
    def __init__(self, task: str) -> None:
        self.task = task

    def __get_keys_to_compare(self, data: Dict, is_point_estimates: bool = False) -> Dict[List[str]]:
        """
        This method gets the keys to calculate the metrics for the given task

        :param data: dictionary with the data to calculate the metrics

        :return list of keys to calculate the metrics
        """
        search_key = "_output"
        point_estimates_fields = ['odds_ratio_output', 'se_log_odds_ratio_output', 'risk_ratio_output', 'se_log_risk_ratio_output', 'mean_difference_output', 'se_mean_difference_output']
        if is_point_estimates:
            relevant_output_fields = [key for key in data.keys() if key in point_estimates_fields]
        else:
            relevant_output_fields = [key for key in data.keys() if search_key in key and key not in point_estimates_fields]
        
        relevant_reference_fields = [key.replace(search_key, "") for key in relevant_output_fields]
        return relevant_output_fields, relevant_reference_fields

    def __calculate_accuracy(self, actual: List[str], predicted: List[str]) -> float:
        """
        This method calculates the accuracy metric

        :param actual: list of actual values
        :param predicted: list of predicted values

        :return accuracy as a float
        """
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual))
    
    def __calculate_mean_absolute_error(self, actual: List[str], predicted: List[str]) -> float:
        """
        This method calculates the mean absolute error

        :param actual: list of actual values
        :param predicted: list of predicted values

        :return: mean absolute error as a float
        """
        return sum([abs(a - p) for a, p in zip(actual, predicted)]) / len(actual)
    
    def __calculate_mean_squared_error(self, actual: List[str], predicted: List[str]) -> float:
        """
        This method calculates the mean squared error

        :param actual: list of actual values
        :param predicted: list of predicted values

        :return: mean squared error as a float
        """
        return sum([(a - p) ** 2 for a, p in zip(actual, predicted)]) / len(actual)
    
    def __calculate_root_mean_squared_error(self, actual: List[str], predicted: List[str]) -> float:
        """
        This method calculates the root mean squared error

        :param actual: list of actual values
        :param predicted: list of predicted values

        :return: root mean squared error as a float
        """
        return self.__calculate_mean_squared_error(actual, predicted) ** 0.5

    def __calculate_exact_match_accuracy(self, data: List[Dict]) -> Dict:
        """
        This method calculates the exact match accuracy for the given task

        :param data: list of dictionaries with the data to calculate the accuracy
        :return: dictionary with the metrics
        """
        item = data[0]
        relevant_output_fields, relevant_reference_fields = self.__get_keys_to_compare(item, False)

        metrics = {}
        total_actual = []
        total_predicted = []
        for output, reference in zip(relevant_output_fields, relevant_reference_fields):
            actual = [example[reference] for example in data]
            predicted = [example[output] for example in data]

            total_actual.extend(actual)
            total_predicted.extend(predicted)

            metrics[reference] = self.__calculate_accuracy(actual, predicted)

        # get the total accuracy
        metrics["total"] = self.__calculate_accuracy(total_actual, total_predicted)
        return metrics
    
    def __calculate_partial_match_accuracy(self, data: List[Dict]) -> Dict:
        """
        This method calculates the partial match accuracy for the given task

        :param data: list of dictionaries with the data to calculate the accuracy
        :return: dictionary with metrics
        """
        # Discussed with Byron and we decided to get different levels of partial matching metrics so this could be 1, 2, 3, 4, 5 etc.
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
            correct = 0
            for example in data:
                num_parts_correct = 0
                for output, reference in zip(relevant_output_fields, relevant_reference_fields):
                    if example[output] == example[reference]:
                        num_parts_correct += 1
                if num_parts_correct >= num_match:
                    correct += 1

            metrics[f"partial_match_accuracy_{num_match}"] = correct / float(len(data))

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
            unknowns = sum([1 for example in data if example[field] == "x" and example[relevant_reference_fields[i]] != "x"])
            metrics[relevant_reference_fields[i]] = unknowns
        metrics["total"] = sum(metrics.values())
        return metrics
    
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
            # calculate the mean squared error
            mean_squared_error = self.__calculate_mean_squared_error(actual, predicted)
            # calculate the root mean squared error
            root_mean_squared_error = self.__calculate_root_mean_squared_error(actual, predicted)
            metrics[reference] = {"mean_absolute_error": mean_absolute_error, "mean_squared_error": mean_squared_error, "root_mean_squared_error": root_mean_squared_error}

        return metrics

    def calculate_metrics(self, data: List[Dict]) -> Dict:
        """
        This method calculates the metrics for the given task

        :param data: list of dictionaries with the data to calculate the metrics
        :return: dictionary with the metrics
        """
        metrics = {}

        # calculate exact match accuracy
        metrics["exact_match_accuracy"] = self.__calculate_exact_match_accuracy(data)
        # calcualte partial match accuracy
        metrics["partial_match_accuracy"] = self.__calculate_partial_match_accuracy(data)

        # calculate number of unknowns
        metrics["number_of_model_unknowns"] = self.__calculate_number_of_model_unknowns(data)

        if self.task == "binary_outcomes" or self.task == "continuous_outcomes":
            # calculate the metrics for point estimates
            metrics["point_estimates"] = self.__calculate_point_estimates_metrics(data)

        return metrics

