import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from datetime import timedelta
import plotly.graph_objects as go

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Neural Network Model
class GapFillerNN(nn.Module):
    def __init__(self, input_size):
        super(GapFillerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

def create_sequences(data, window_size):
    """Create sequences for the rolling window approach"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Pretraining Function
def pretrain_nn(model, file_path, scaler_y, window_size=200, epochs=30000):
    try:
        # Load and preprocess external data
        df_ext = pd.read_csv(file_path, parse_dates=['timestamp'])
        df_ext.set_index('timestamp', inplace=True)
        
        # Scale data
        data = df_ext['energy_mwh'].values
        data_scaled = scaler_y.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train, y_train = create_sequences(data_scaled, window_size)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        print(f"Pretraining complete with file: {file_path}")
        return model

    except Exception as e:
        print(f"Error during pretraining: {e}")
        raise

# Fill gaps with Neural Networks
def fill_gap_with_nn(df_reindexed, start_gap, end_gap, model, scaler_y, window_size=200):
    try:
        # Get data before the gap
        df_before_gap = df_reindexed[:start_gap].dropna()
        
        if len(df_before_gap) < window_size:
            raise ValueError(f"Not enough data before the gap to create a rolling window of size {window_size}")
        
        # Scale the data
        data_before = df_before_gap['energy_mwh'].values
        data_scaled = scaler_y.transform(data_before.reshape(-1, 1)).flatten()
        
        # Predict dynamically across the gap
        current_time = start_gap
        while current_time <= end_gap:
            # Get the last window_size values
            last_window = data_scaled[-window_size:]
            
            # Convert to tensor and predict
            model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)
                prediction_scaled = model(input_tensor).item()
            
            # Scale back the prediction
            prediction = scaler_y.inverse_transform([[prediction_scaled]])[0, 0]
            
            # Assign the prediction to the gap
            df_reindexed.loc[current_time, 'energy_mwh'] = prediction
            
            # Update the scaled data array with the new prediction
            data_scaled = np.append(data_scaled, prediction_scaled)
            
            # Increment time
            current_time += timedelta(hours=1)
            
    except Exception as e:
        print(f"Error filling gap from {start_gap} to {end_gap}: {e}")
        raise

# Main Code
def main():
    # Load the CSV data
    df = pd.read_csv('report_admie_realtimescadasystemload_2024-01-01_2024-11-22.csv', parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Initialize scaler
    scaler_y = MinMaxScaler()

    # Create and pretrain the model
    window_size = 200  # Using 200 hours as input
    model = GapFillerNN(input_size=window_size).to(device)
    
    try:
        model = pretrain_nn(model, 'external_energy_data.csv', scaler_y, window_size=window_size, epochs=500)

        # Fill gaps in the main dataset
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df_reindexed = df.reindex(full_date_range)
        gaps = df_reindexed.index[df_reindexed['energy_mwh'].isna()]

        if len(gaps) > 0:
            gap_starts = [gaps[0]]
            gap_ends = []
            
            for i in range(1, len(gaps)):
                if gaps[i] - gaps[i-1] > pd.Timedelta(hours=1):
                    gap_ends.append(gaps[i-1])
                    gap_starts.append(gaps[i])
            gap_ends.append(gaps[-1])

            for start_gap, end_gap in zip(gap_starts, gap_ends):
                print(f"Filling gap from {start_gap} to {end_gap}")
                fill_gap_with_nn(df_reindexed, start_gap, end_gap, model, scaler_y, window_size=window_size)

        # Save the filled data
        df_reindexed.to_csv('filled_energy_data_nn_cuda.csv')
        print("All gaps filled and saved to 'filled_energy_data_nn_cuda.csv'")

        # Plot the original and filled data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_reindexed.index, y=df_reindexed['energy_mwh'], 
                               mode='lines', name='Filled Data'))
        fig.add_trace(go.Scatter(x=df.index, y=df['energy_mwh'], 
                               mode='markers', name='Original Data', marker=dict(color='red')))
        fig.update_layout(title="Energy Data Gap Filling", 
                         xaxis_title="Timestamp", 
                         yaxis_title="Energy (MWh)")
        fig.show()

    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()