import random
import csv
import math
from scipy.stats import binomtest
import numpy as np

class Node:
    def __init__(self, node_type):
        self.node_type = node_type  # 1 for normal, 0 for anomaly
        if node_type == 1:
            # Normal nodes have a 3% chance of error
            self.rand_number = 0.03
        else:
            # Anomaly nodes have a random chance of error between 25% and 80%
            self.rand_number = random.uniform(0.25, 0.8)

    def run_operation(self):
        return random.random() >= self.rand_number  

    def get_node_type(self):
        return self.node_type
    
    def get_node_odds(self):
        return self.rand_number

class Server:
    def __init__(self, num_nodes, operations_per_node):
        self.operations_per_node = operations_per_node
        self.num_nodes = num_nodes
        self.nodes = []
        self.results = {}

        for _ in range(num_nodes):
            # 50% chance to be an anomaly node
            node_type = 0 if random.random() < 0.5 else 1
            self.nodes.append(Node(node_type))

    def execute(self):
        for node_index, node in enumerate(self.nodes):
            self.results[node_index] = [1 if node.run_operation() else 0 for _ in range(self.operations_per_node)]

    def get_results(self):
        return self.results
    
    def get_num_nodes(self):
        return self.num_nodes

def perform_binomial_test(server, significance_level=0.05, one_tailed=True):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for node_index, node_results in server.get_results().items():
        num_successes = sum(node_results)
        num_trials = len(node_results)
        expected_prob = server.nodes[node_index].get_node_odds()

        # Adjust for one-tailed test if necessary
        if one_tailed:
            significance_level /= 2

        test_result = binomtest(num_successes, num_trials, 1 - expected_prob)
        p_value = test_result.pvalue

        actual_anomaly = server.nodes[node_index].get_node_type() == 0
        detected_anomaly = (p_value < significance_level) if one_tailed else (p_value / 2 < significance_level)

        if actual_anomaly and detected_anomaly:
            true_positives += 1
        elif not actual_anomaly and detected_anomaly:
            false_positives += 1
        elif not actual_anomaly and not detected_anomaly:
            true_negatives += 1
        elif actual_anomaly and not detected_anomaly:
            false_negatives += 1

        print(f"Node {node_index}: Actual: {'Anomaly' if actual_anomaly else 'Normal'}, "
              f"Detected: {'Anomaly' if detected_anomaly else 'Normal'} (p-value: {p_value})")

    success_rate = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives) if server.get_num_nodes() > 0 else 0
    print(f"\nTrue Positives: {true_positives}, False Positives: {false_positives}, "
          f"True Negatives: {true_negatives}, False Negatives: {false_negatives}, "
          f"Success Rate: {success_rate:.2f}")

def perform_z_score_test(server, expected_success_rate=0.97, threshold=-1.96):
    results = server.get_results()
    # Standard deviation for the expected success rate
    std_dev = math.sqrt(expected_success_rate * (1 - expected_success_rate))

    # Calculate z-scores and determine anomalies
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for node_index, node_results in results.items():
        mean_success_rate = sum(node_results) / len(node_results)
        z_score = (mean_success_rate - expected_success_rate) / std_dev
        detected_anomaly = z_score < threshold
        actual_anomaly = server.nodes[node_index].get_node_type() == 0

        # Increment counters based on the actual and detected anomaly
        if actual_anomaly and detected_anomaly:
            true_positives += 1
        elif not actual_anomaly and detected_anomaly:
            false_positives += 1
        elif not actual_anomaly and not detected_anomaly:
            true_negatives += 1
        elif actual_anomaly and not detected_anomaly:
            false_negatives += 1

        print(f"Node {node_index}: Actual: {'Anomaly' if actual_anomaly else 'Normal'}, "
              f"Detected: {'Anomaly' if detected_anomaly else 'Normal'} (Z-Score: {z_score})")

    # Calculate and print the success rate
    total_nodes = true_positives + false_positives + true_negatives + false_negatives
    success_rate = (true_positives + true_negatives) / total_nodes if total_nodes > 0 else 0
    print(f"\nTrue Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Success Rate: {success_rate:.2f}")

def detect_outliers_iqr(data):
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers

def perform_chi_squared_test(server, observed, expected):
    results=server.get_results()
    # Calculate the chi-squared test statistic
    chi_squared = 0

    for node_index, node_results in results.items():
        expected=[1,1,1,1,1,1]
        observed=server.getresults[node_index]
        for obs, exp in zip(observed, expected):
            chi_squared += ((obs - exp) ** 2) / exp

    # Calculate the degrees of freedom
    degrees_of_freedom = len(observed) - 1

    return chi_squared,degrees_of_freedom

def main():
    num_nodes = 100
    operations_per_node = 1000

    server = Server(num_nodes, operations_per_node)
    server.execute()
    # save the results
    results = server.get_results()

    # Perform tests and print results
    print("Performing Z Test")
    perform_z_score_test(server)

    print("\nPerforming Binomial Test")
    perform_binomial_test(server)

    outliers = detect_outliers_iqr(data)
    print("Detected outliers:", outliers)

    """"
    # Writing a separate CSV for anomaly status
    with open('node_anomalies.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node Index', 'Success Rate', 'Z-Score', 'Anomaly Status'])
        for node in range(num_nodes):
            writer.writerow([
                node, 
                sum(results[node]) / operations_per_node, 
                z_scores[node], 
                "Yes" if z_scores[node] < threshold else "No"
            ])
    """

if __name__ == "__main__":
    main()

