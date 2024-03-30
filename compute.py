import random
import csv
import pandas
import math
import numpy

class Node:
    def __init__(self, node_type):
        self.node_type = node_type # if node_type = 1 then normal, if 0 then anomaly
        if (node_type == 1):
            # normal nodes have a 3% chance of error
            self.rand_number = 0.03
        else:
            # anomaly nodes have a random chance of error between 25 and 80%
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
        anomaly_chance = 0.05
        self.nodes = []
        self.results = {}  # Initialize an empty dictionary for results

        # Create nodes and store them in a list
        for _ in range(num_nodes):
            node_type = 0 if random.random() < anomaly_chance else 1
            self.nodes.append(Node(node_type))

    def execute(self):
        for node_index, node in enumerate(self.nodes):
            if node_index not in self.results:
                self.results[node_index] = []

            for _ in range(self.operations_per_node):
                operation_result = 1 if node.run_operation() else 0
                self.results[node_index].append(operation_result)     
    
    def get_results(self):
        return self.results
    
    def return_anomalies(self):
        anomalies = []
        for node_index, node in enumerate(self.nodes):
            if node.get_node_type() == 0:
                anomalies.append(node_index)
        return anomalies

  
def calculate_z_scores(data):
    # Calculate mean and standard deviation
    means = {node: sum(ops) / len(ops) for node, ops in data.items()}
    std_devs = {
        node: math.sqrt(sum((x - means[node]) ** 2 for x in ops) / len(ops))
        for node, ops in data.items()
    }
    
    # Calculate z-scores (number of standard deviations from the mean)
    z_scores = {
        node: (means[node] - 0.5) / std_devs[node] if std_devs[node] else float('inf')
        for node in data
    }
    
    return z_scores

def main():
    num_nodes = 10
    operations_per_node = 1000
    #initialize server with 10 nodes and 10 operations per node
    server = Server(num_nodes, operations_per_node)
    # execute the operations of each node in the server
    server.execute()
    # save the results
    results = server.get_results()
    
    # Open a file for writing
    with open('data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row based on the number of nodes
        headers = [f'Node {i}' for i in range(num_nodes)]
        writer.writerow(headers)
        
        # Transpose the results to write them as columns
        for operation_index in range(operations_per_node):
            row = [results[node_index][operation_index] for node_index in range(num_nodes)]
            writer.writerow(row)

    
    # Calculate the z-scores for each node
    z_scores = calculate_z_scores(results)
    
    # Decide on a threshold and print out anomalies
    threshold = 1.96  # corresponds to 95% confidence interval
    anomalies = {node: score for node, score in z_scores.items() if abs(score) > threshold}
    
    print("Anomalies based on z-scores:")
    for node, z_score in anomalies.items():
        print(f"Node {node} with z-score {z_score}")

    print(server.return_anomalies())



if __name__ == "__main__":
    main()
        