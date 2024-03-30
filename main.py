import math
import csv
import random

# ... (Node and Server class definitions remain unchanged)

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
    operations_per_node = 100  # More operations for a meaningful statistical test
    
    # Initialize and run the server
    server = Server(num_nodes, operations_per_node)
    server.execute()
    results = server.get_results()
    
    # Calculate the z-scores for each node
    z_scores = calculate_z_scores(results)
    
    # Decide on a threshold and print out anomalies
    threshold = 1.96  # corresponds to 95% confidence interval
    anomalies = {node: score for node, score in z_scores.items() if abs(score) > threshold}
    
    print("Anomalies based on z-scores:")
    for node, z_score in anomalies.items():
        print(f"Node {node} with z-score {z_score}")

if __name__ == "__main__":
    main()
