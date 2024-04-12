import random
import csv
import math
from scipy.stats import binomtest
import numpy as np
import pandas as pd
#travis imports


from scipy.stats import chisquare

# Machine Learning Imports
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l1_l2

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




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
            node_type = 0 if random.random() < 0.1 else 1
            self.nodes.append(Node(node_type))

    def execute(self):
        self.results.clear()
        for node_index, node in enumerate(self.nodes):
            self.results[node_index] = [1 if node.run_operation() else 0 for _ in range(self.operations_per_node)]

    def get_results(self):
        return self.results
    
    def get_num_nodes(self):
        return self.num_nodes
    
    def get_num_operations(self):
        return self.operations_per_node
    
    def prune_malicious_nodes(self, malicious_list):
        # Ensure unique indices to avoid double removal attempts
        unique_indices = set(malicious_list)
        
        # Sort indices in descending order to avoid index shifting issues
        for i in sorted(unique_indices, reverse=True):
            # Check if the index is within the current range of the list
            if 0 <= i < len(self.nodes):
                self.nodes.pop(i)
    
    def write_csv(self, z_score, binom, isol, chi, iqr):
        # Writing a separate CSV for anomaly status
        with open('node_anomalies.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Node Index', 'Actual', 'Z-Score', 'Binomial', 'Isolation Forest', 'Chi Squared', 'IQR'])
            for node_index in range(self.num_nodes):
                writer.writerow([
                    node_index, 
                    self.nodes[node_index].get_node_type(), 
                    z_score[1][node_index], 
                    binom[1][node_index],
                    isol[1][node_index], 
                    chi[1][node_index],
                    iqr[1][node_index]
                ])
                
        with open('overall_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Test', 'True Positive', 'False Positive', 'True Negative', 'False Negative'])

            tests = [z_score, binom, isol, chi, iqr]

            for i in range(5):
                writer.writerow([
                    tests[i][0], 
                    tests[i][2],
                    tests[i][3],
                    tests[i][4],
                    tests[i][5]
                ])
        

def perform_ML_test(server, epochs=100, batch_size=2, threshold_quantile=0.9):
        # Convert results to DataFrame for easier processing
        data = pd.DataFrame(server.get_results()).T
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Define the autoencoder network architecture
        input_dim = scaled_data.shape[1]
        encoding_dim = 8  # Increased from 4 to 8

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="relu")(input_layer)
        encoder = Dense(encoding_dim // 2, activation="relu")(encoder)
        # Adding an extra layer to make the model deeper
        encoder = Dense(encoding_dim // 4, activation="relu")(encoder)
        encoder = Dense(encoding_dim, activation="relu", activity_regularizer=l1_l2(l1=0.0001, l2=0.0001))(input_layer)


        decoder = Dense(encoding_dim // 4, activation="relu")(encoder)
        decoder = Dense(encoding_dim // 2, activation="relu")(decoder)
        decoder = Dense(input_dim, activation="sigmoid")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

        autoencoder.fit(scaled_data, scaled_data,
                        epochs=200,  # Increased epochs
                        batch_size=10,  # Increased batch size
                        shuffle=True,
                        validation_split=0.2,
                        verbose=1)

        # Predict and calculate the reconstruction error
        reconstructed_data = autoencoder.predict(scaled_data)
        reconstruction_error = np.mean(np.abs(scaled_data - reconstructed_data), axis=1)

        # Determine the threshold for anomaly detection
        threshold = np.quantile(reconstruction_error, 0.95)  # Adjust the quantile based on error distribution
        
        # Detect anomalies
        anomalies = reconstruction_error > threshold
        anomaly_indices = np.where(anomalies)[0]

        # Evaluate the detection
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for i, node in enumerate(server.nodes):
            actual_anomaly = node.get_node_type() == 0
            detected_anomaly = i in anomaly_indices

            if actual_anomaly and detected_anomaly:
                true_positives += 1
            elif not actual_anomaly and detected_anomaly:
                false_positives += 1
            elif not actual_anomaly and not detected_anomaly:
                true_negatives += 1
            elif actual_anomaly and not detected_anomaly:
                false_negatives += 1

            print(f"Node {i}: Actual: {'Anomaly' if actual_anomaly else 'Normal'}, "
                  f"Detected: {'Anomaly' if detected_anomaly else 'Normal'}")

        success_rate = (true_positives + true_negatives) / (server.num_nodes) if server.num_nodes > 0 else 0
        print(f"\nTrue Positives: {true_positives}, False Positives: {false_positives}, "
              f"True Negatives: {true_negatives}, False Negatives: {false_negatives}, "
              f"Success Rate: {success_rate:.2f}")

        return anomaly_indices

def perform_z_score_test(server, expected_success_rate=0.97, threshold=-1.5):
    results = server.get_results()
    # Standard deviation for the expected success rate
    std_dev = math.sqrt(expected_success_rate * (1 - expected_success_rate))

    # Calculate z-scores and determine anomalies
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # initialize list of anomaly nodes
    nodes = []
    for node_index, node_results in results.items():
        mean_success_rate = sum(node_results) / len(node_results)
        z_score = (mean_success_rate - expected_success_rate) / std_dev
        detected_anomaly = z_score < threshold
        actual_anomaly = server.nodes[node_index].get_node_type() == 0

        # if the node is detected as an anomaly add it to the list
        if detected_anomaly:
            nodes.append(1)
        else:
            nodes.append(0)


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

    return "Z-Score", nodes, true_positives, false_positives, true_negatives, false_negatives

def perform_binomial_test(server, significance_level=0.05, one_tailed=True):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    nodes = []
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

        # if the node is detected as an anomaly add it to the list
        if detected_anomaly:
            nodes.append(1)
        else:
            nodes.append(0)

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
    
    return "Binomial", nodes, true_positives, false_positives, true_negatives, false_negatives

"""""""""""""""
def return_anomalies_iqr(self, server,  multiplier=1.5):
    actual_anomaly = server.nodes[node_index].get_node_type() == 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Calculate success rates for each node
    success_rates = [sum(self.results[node]) / len(self.results[node]) for node in self.results]

    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(success_rates, 25)
    Q3 = np.percentile(success_rates, 75)
    IQR = Q3 - Q1

    # Determine lower and upper bounds for anomalies
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Identify anomalies
    anomalies = [i for i, rate in enumerate(success_rates) if rate < lower_bound or rate > upper_bound]
    
    nodes = []

    if anomalies:
        nodes.append(1)
    else:
        nodes.append(0)
    
    if actual_anomaly and detected_anomaly:
        true_positives += 1
    elif not actual_anomaly and detected_anomaly:
        false_positives += 1
    elif not actual_anomaly and not detected_anomaly:
        true_negatives += 1
    elif actual_anomaly and not detected_anomaly:
        false_negatives += 1

    return nodes
    """

def perform_chi_squared_test(server):

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    results=server.get_results()
    expected = [[1 for i in range(server.get_num_operations())] for j in range(server.get_num_nodes())]

    keys = list(results.keys())
    values = list(results.values())

    # Create a 2D NumPy array
    numpy_array = np.array(values)

    outdata=np.zeros(len(numpy_array))
    size=len(numpy_array)

    # Perform chi-squared test
    for i, size in enumerate(numpy_array):

        res = chisquare(f_obs=numpy_array[i])
        print("Chi-squared Statistic ", i, ":" ,res.statistic)
        #print("P-value:", res.pvalue)
        if (res.statistic>100):
            outdata[i]=1
            #print("Anomaly detected at node ", i)

        #true positive
        if (res.statistic>100 and server.nodes[i].get_node_type() == 0):
            true_positives+=1
        elif (res.statistic>100 and server.nodes[i].get_node_type() == 1):
            false_positives+=1
        elif (res.statistic<100 and server.nodes[i].get_node_type() == 0):
            false_negatives+=1
        elif (res.statistic<100 and server.nodes[i].get_node_type() == 1):
            true_negatives+=1

    total_nodes = true_positives + false_positives + true_negatives + false_negatives
    success_rate = (true_positives + true_negatives) / total_nodes if total_nodes > 0 else 0
    print(f"\nTrue Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Success Rate: {success_rate:.2f}")

    return "Chi-Squared", outdata,true_negatives,true_positives,false_negatives,false_positives,success_rate
def apply_isolation_forest(server):
    # Preparing data with true labels
    data = {
        "node_index": [],
        "success_rate": [],
        "true_label": []  # Add a column for true labels
    }
    for node_index, node in enumerate(server.nodes):
        operations = server.results[node_index]
        success_rate = sum(operations) / len(operations)
        data["node_index"].append(node_index)
        data["success_rate"].append(success_rate)
        # Assuming node_type 1 is normal (label as 0) and 0 is anomaly (label as 1)
        data["true_label"].append(0 if node.get_node_type() == 1 else 1)

    df = pd.DataFrame(data)

    # Applying Isolation Forest with a contamination factor of 0.1
    isolation_forest = IsolationForest(n_estimators=100, random_state=42, contamination=0.01)
    predictions = isolation_forest.fit_predict(df[['success_rate']])
    # Transform predictions: -1 (anomaly) becomes 1, 1 (normal) becomes 0
    df['is_anomaly'] = [1 if x == -1 else 0 for x in predictions]

    # Extracting the true labels and predictions
    true_labels = df['true_label']
    anomaly_predictions = df['is_anomaly']

    # Calculating confusion matrix components and accuracy
    tn, fp, fn, tp = confusion_matrix(true_labels, anomaly_predictions).ravel()
    accuracy = accuracy_score(true_labels, anomaly_predictions)

    metrics = {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Accuracy': accuracy
    }

    # Print the confusion matrix components
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Anomaly detection accuracy: {accuracy:.2f}")

def main():
    num_nodes = 100
    operations_per_node = 1000

    server = Server(num_nodes, operations_per_node)
    server.execute()
    # save the results
    results = server.get_results()

    # Perform tests and print results
    print("Performing Z Test")
    nodes_z = perform_z_score_test(server)

    #print("Performing ML Anomaly Detection")
    #node_ml = perform_ML_test(server)
    
    print("\nPerforming Binomial Test")
    nodes_binom = perform_binomial_test(server)

    print("\nPerforming IQR Test")
    #iqr_anomalies = return_anomalies_iqr(server, 1.25)

    print("\nPerforming Chi-Square Test")
    chisquare_anomalies=perform_chi_squared_test(server)

    print("Performing Isolation Forrest Classification")
    isolation_forest = apply_isolation_forest(server)

    server.write_csv(nodes_z, nodes_binom, isolation_forest, chisquare_anomalies, nodes_binom)


if __name__ == "__main__":
    main()

