import numpy as np
import pandas as pd
from scipy.optimize import linprog

class TrafficManagementSystem:
    def __init__(self, num_lanes):
        self.num_lanes = num_lanes
        self.flow_data = np.zeros((num_lanes, num_lanes))  # Matrix to hold traffic flow data

    def set_traffic_flow(self, source_lane, destination_lane, flow):
        self.flow_data[source_lane][destination_lane] = flow

    def load_data_from_csv(self, csv_file):
        data = pd.read_csv(csv_file)
        for _, row in data.iterrows():
            self.set_traffic_flow(int(row['Source Lane']), int(row['Destination Lane']), row['Flow'])

    def optimize_traffic_flow(self):
        # Objective function: Minimize total flow
        c = np.ones(self.num_lanes)
        
        # Constraints: sum of flow should be <= 1 for each lane
        A_ub = np.vstack([np.eye(self.num_lanes), -np.eye(self.num_lanes)])
        b_ub = np.hstack([np.ones(self.num_lanes), np.zeros(self.num_lanes)])
        
        # Solve the linear programming problem
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        
        if result.success:
            print("Optimized Traffic Flow:", result.x)
        else:
            print("Optimization failed!")

if __name__ == "__main__":
    system = TrafficManagementSystem(num_lanes=4)
    
    # Load data from CSV file
    system.load_data_from_csv('data_example.csv')
    
    # Optimize traffic flow
    system.optimize_traffic_flow()
