import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('job_results.db')
cursor = conn.cursor()

# Query to select all data from the 'results' table
query = "SELECT * FROM results"
cursor.execute(query)

# Fetch all rows from the query
rows = cursor.fetchall()

# Print the rows
print("Host, Param, Anomaly Type, CPU Load, Memory Usage, Disk I/O, Network Latency")
for row in rows:
    print(', '.join(str(item) for item in row))

# Close the database connection
conn.close()