# Main Theme: Simulating login traffic and applying machine learning (Isolation Forest) to detect security anomalies

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# =========================================================================
# TODO 1 & 2: Log Collection, Parsing & Normalizing (Optional/Mocked for now)
# =========================================================================
# This section is commented out, but shows how you would normally load
# and structure real log data from a JSON file into a Pandas DataFrame.
#
# data = pd.read_json("logs_15m.json")
# timestamp = data["timestamp"]
# process = data["processID"]
# message = data["eventMessage"]
# user = data["userID"]
#
# new_df = pd.DataFrame({
#     "Time_Logged": timestamp,
#     "Process_ID": process,
#     "Event_Details": message,
#     "User_ID": user
# })


# =========================================================================
# TODO: Derive Features for Anomaly Detection (Simulated Data)
# =========================================================================
# Instead of a real log file, we simulate data to test our machine learning model.
np.random.seed(42)  # Sets a seed so the random numbers are the same every time we run it

# Simulate 50 minutes of normal login traffic (averaging 5 attempts per minute)
normal_counts = np.random.poisson(lam=5, size=50)

# Simulate an anomaly: a sudden security spike (e.g., 30, 40, and 50 login attempts in a minute)
anomalous_counts = np.array([30, 40, 50])

# Combine normal traffic and spikes into one complete dataset array
login_attempts = np.concatenate([normal_counts, anomalous_counts])
print("Login attempts per minute:", login_attempts)


# =========================================================================
# TODO 3: Analysis and Detection (Machine Learning Approach)
# =========================================================================
# Reshape the 1D flat list into a 2D column array (samples, features)
# because scikit-learn models expect 2D inputs.
X = login_attempts.reshape(-1, 1)

# Initialize the Isolation Forest anomaly detection model.
# contamination=0.05 tells the model we expect about 5% of the data points to be anomalies.
model = IsolationForest(contamination=0.05, random_state=42)

# Train (fit) the model on our login attempt data so it learns what "normal" looks like.
model.fit(X)


# =========================================================================
# TODO 4: Dashboard / Visualization
# =========================================================================
# Use the trained model to make predictions.
# It returns +1 for normal data points and -1 for anomalies.
labels = model.predict(X)

# Find the exact positions (indices) and values of the flagged anomalies
anomaly_indices = np.where(labels == -1)[0]
anomaly_values = login_attempts[anomaly_indices]

print("Anomaly indices:", anomaly_indices)
print("Anomaly values (login attempts):", anomaly_values)

# Plot the login traffic on a line graph and highlight anomalies in red on a scatter plot
plt.plot(login_attempts, label="Login attempts per minute")
plt.title("Login Attempts Over 53 Minutes")
plt.scatter(anomaly_indices, anomaly_values, color='red', label="Anomalies")
plt.xlabel("Time (minute index)")
plt.ylabel("Login attempts")
plt.legend()
plt.show()