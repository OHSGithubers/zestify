import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


# Extract percent data
def extract_percents(data):
    # Initialize percents
    data_percents = []

    # Loop and generate percent changes
    for i in range(data.shape[0]):
        try:
            data_close1 = data["Close"][i]
            i3 = i+1
            data_close2 = data["Close"][i3]
            data_percent_var = (data_close1 - data_close2)/data_close2
            data_percents.append(data_percent_var)
        except:
            pass

    # Return percents
    return data_percents


# Load data
stock_data = pd.read_csv("stonk_data.csv")

# Load percents and remove last entry
extracted_percent_data = extract_percents(stock_data)
stock_data = stock_data[:-1]
stock_data["Percents"] = extracted_percent_data

# Reverse data
stock_data = stock_data[::-1].reset_index(drop=True)

# Convert date info to decimal form
# stock_data["Date"] = list(map(lambda t: time.mktime(date.fromisoformat(t).timetuple()), stock_data["Date"]))
# Remove date info
stock_data = stock_data.drop("Date", axis=1)

# Scale and separate data
scaler = StandardScaler()
processed_data = scaler.fit_transform(stock_data.drop("Close", axis=1).values)

# Add raw Close data back to dataset for sequencing
processed_data = np.column_stack((processed_data, stock_data["Close"].values))

# Sequence extractor
def extract_sequences(data, step_days=50):
    # Empty sequence list
    sequences = []

    # Loop array and step
    for i in range(len(data)):
        # Check for end of window and array
        if i + step_days + 1 > len(data):
            break

        # Add sequence
        sequences.append(data[i:i+step_days])

    # Return sequences
    return np.asarray(sequences)

# Extract sequences
sequence_data = extract_sequences(processed_data)

# Derive result data
X_data = sequence_data
y_data = stock_data["Close"].values[50:]

# Save data
with open("processed_stonk_data", "wb") as f:
    pickle.dump((X_data, y_data), f)
