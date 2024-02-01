from typing import Dict, List

class MetricsCalculator:
    def __init__(self, task: str) -> None:
        self.task = task

    def __get_keys_to_compare(data: Dict) -> Dict[List[str]]:
        """
        This method gets the keys to calculate the metrics for the given task

        :param data: dictionary with the data to calculate the metrics

        :return list of keys to calculate the metrics
        """
        search_key = "_output"
        relevant_output_fields = [key for key in data.keys() if search_key in key]
        relevant_reference_fields = [key.replace(search_key, "") for key in relevant_output_fields]
        return relevant_output_fields, relevant_reference_fields

    def __accuracy_metric(actual: List[str], predicted: List[str]) -> float:
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

    def __calculate_exact_match_accuracy(self, data: List[Dict]) -> Dict:
        """
        This method calculates the exact match accuracy for the given task

        :param data: list of dictionaries with the data to calculate the accuracy
        :return: dictionary with the metrics
        """
        item = data[0]
        relevant_output_fields, relevant_reference_fields = self.__get_keys_to_compare(item)

        metrics = {}
        total_actual = []
        total_predicted = []
        for output, reference in zip(relevant_output_fields, relevant_reference_fields):
            actual = [example[reference] for example in data]
            predicted = [example[output] for example in data]

            total_actual.extend(actual)
            total_predicted.extend(predicted)

            metrics[reference] = self.__accuracy_metric(actual, predicted)

        # get the total accuracy
        metrics["total"] = self.__accuracy_metric(total_actual, total_predicted)
        return metrics
    
    def __calculate_partial_match_accuracy(self, data: List[Dict]) -> float:
        """
        This method calculates the partial match accuracy for the given task

        :param data: list of dictionaries with the data to calculate the accuracy
        :return: dictionary with metrics
        """
        if self.task == "binary_outcomes":
            num_match = 2
        elif self.task == "continuous_outcomes":
            num_match = 3
        else:
            num_match = 1

        item = data[0]
        relevant_output_fields, relevant_reference_fields = self.__get_keys_to_compare(item)

        correct = 0
        for example in data:
            for output, reference in zip(relevant_output_fields, relevant_reference_fields):
                num_parts_correct = 0
                if example[output] == example[reference]:
                    num_parts_correct += 1
            if num_parts_correct >= num_match:
                correct += 1

        return correct / float(len(data))

    # Does it even make sense to do precision, recall, f1 for this type of task?
    # def __calculate_exact_match_precision_recall_f1(self, data: List[Dict]) -> Dict:
    #     """
    #     This method calculates the exact match precision, recall, f1

    #     :param data: list of dictionaries with the data to calculate the metrics
    #     :return: dictionary with the metrics
    #     """
    #     return {}

    # def __calculate_partial_match_precision_recall_f1(self, data: List[Dict]) -> Dict:
    #     """
    #     This method calculates the partial match precision, recall, f1

    #     :param data: list of dictionaries with the data to calculate the metrics
    #     :return: dictionary with the metrics
    #     """
    #     return {}

    def __calculate_number_of_model_unknowns(self, data: List[Dict]) -> Dict:
        """
        This method calculates the number of unknowns or "x" in the data only given by the model

        :param data: list of dictionaries with the data to calculate the number of unknowns
        :return: Dictionary with the number of unknowns for each field and total
        """
        item = data[0]
        relevant_output_fields, relevant_reference_fields = self.__get_keys_to_compare(item)
        
        metrics = {}
        for i, field in enumerate(relevant_output_fields):
            unknowns = sum([1 for example in data if example[field] == "x" and example[relevant_reference_fields[i]] != "x"])
            metrics[relevant_reference_fields[i]] = unknowns
        metrics["total"] = sum(metrics.values())
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

        # calculate exact match precision, recall, and f1
        # exact_match_metrics = self.__calculate_exact_match_precision_recall_f1(data)
        # metrics["exact_match_precision"] = exact_match_metrics["precision"]
        # metrics["exact_match_recall"] = exact_match_metrics["recall"]
        # metrics["exact_match_f1"] = exact_match_metrics["f1"]

        # calculate partial match precision, recall, and f1
        # partial_match_metrics = self.__calculate_partial_match_precision_recall_f1(data)
        # metrics["partial_match_precision"] = partial_match_metrics["precision"]
        # metrics["partial_match_recall"] = partial_match_metrics["recall"]
        # metrics["partial_match_f1"] = partial_match_metrics["f1"]

        # calculate number of unknowns
        metrics["number_of_model_unknowns"] = self.__calculate_number_of_model_unknowns(data)

        return metrics

