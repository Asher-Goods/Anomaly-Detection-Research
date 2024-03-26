import random
import time
import socket
import dispy
import sqlite3

def compute(n):
    host = socket.gethostname()
    cpu_load = random.randint(10, 50)
    memory_usage = random.randint(200, 500)
    disk_io = random.randint(10, 50)
    network_latency = random.randint(10, 50)  # Simulated network latency in milliseconds

    anomaly_type = None
    if random.random() < 0.2:
        anomaly_type = random.choice(['high_cpu', 'high_memory', 'disk_io', 'network_latency'])
        if anomaly_type == 'high_cpu':
            cpu_load = random.randint(50, 100)
        elif anomaly_type == 'high_memory':
            memory_usage = random.randint(500, 1000)
        elif anomaly_type == 'disk_io':
            disk_io = random.randint(50, 100)
        elif anomaly_type == 'network_latency':
            network_latency = random.randint(50, 100)  # Increased network latency in case of anomaly

    # Sleep time adjusted to ensure total execution time remains within limits
    time.sleep(min(n, 10 - network_latency / 1000.0))
    
    return host, n, anomaly_type, cpu_load, memory_usage, disk_io, network_latency

if __name__ == '__main__':
    conn = sqlite3.connect('job_results.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            host TEXT,
            param INTEGER,
            anomaly_type TEXT,
            cpu_load INTEGER,
            memory_usage INTEGER,
            disk_io INTEGER,
            network_latency INTEGER
        )
    ''')
    conn.commit()

    cluster = dispy.JobCluster(compute)
    jobs = []

    # Ensure jobs can complete within the desired time frame
    for i in range(10):
        job = cluster.submit(random.randint(1, 3))  # Adjust job parameters to fit within time constraints
        job.id = i
        jobs.append(job)

    cluster.wait()  # Wait for all jobs to complete

    for job in jobs:
        try:
            host, n, anomaly, cpu_load, memory_usage, disk_io, network_latency = job()
            print(f'Node: {host}, Job: {job.id}, Param: {n}, Anomaly: {anomaly}')
            print(f'CPU Load: {cpu_load}%, Memory Usage: {memory_usage}MB, Disk I/O: {disk_io}MB, Network Latency: {network_latency}ms')

            # Insert job results into the database
            cursor.execute('''
                INSERT INTO results (host, param, anomaly_type, cpu_load, memory_usage, disk_io, network_latency)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (host, n, anomaly, cpu_load, memory_usage, disk_io, network_latency))
            conn.commit()

        except Exception as e:
            print(f'Job {job.id} failed with exception: {e}')

    cluster.print_status()
    conn.close()
