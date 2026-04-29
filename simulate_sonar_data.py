import numpy as np
import csv
import os

def simulate_sonar_data(num_steps=150, dt=1.0, clutter_rate=100, output_csv='sonar_simulation.csv'):
    """
    Simulates a highly maneuverable target in a noisy harbor environment.
    Generates target measurements and random clutter, saving the results to a CSV.
    """
    np.random.seed(42)
    
    # Target initial state: [x, y, vx, vy] in meters and m/s
    # Start at x=1000, y=1000, speed=2 m/s (approx 4 knots)
    x = 1000.0
    y = 1000.0
    vx = 2.0
    vy = 0.0
    
    records = []
    
    # Harbor boundaries for clutter generation
    min_range = 100.0
    max_range = 5000.0
    min_bearing = -np.pi
    max_bearing = np.pi
    
    # Measurement noise standard deviations
    sigma_range = 5.0 # 5 meters range resolution
    sigma_bearing = np.deg2rad(1.0) # 1 degree bearing resolution
    
    print(f"Starting simulation for {num_steps} time steps...")
    
    for t in range(num_steps):
        # Maneuver schedule (Coordinated Turn model logic)
        if 30 <= t < 60:
            # Turn left at 5 degrees per second
            omega = np.deg2rad(5.0) 
        elif 90 <= t < 110:
            # Turn right at 8 degrees per second
            omega = np.deg2rad(-8.0)
        else:
            # Straight line (Constant Velocity)
            omega = 0.0 
            
        # Update true target state using Coordinated Turn or Constant Velocity kinematics
        if abs(omega) > 1e-6:
            x_new = x + (vx / omega) * np.sin(omega * dt) - (vy / omega) * (1 - np.cos(omega * dt))
            y_new = y + (vx / omega) * (1 - np.cos(omega * dt)) + (vy / omega) * np.sin(omega * dt)
            vx_new = vx * np.cos(omega * dt) - vy * np.sin(omega * dt)
            vy_new = vx * np.sin(omega * dt) + vy * np.cos(omega * dt)
            x, y, vx, vy = x_new, y_new, vx_new, vy_new
        else:
            x += vx * dt
            y += vy * dt
            
        # Generate true measurement (range, bearing)
        true_range = np.sqrt(x**2 + y**2)
        true_bearing = np.arctan2(y, x)
        
        # Add measurement noise for the target hit
        meas_range = true_range + np.random.normal(0, sigma_range)
        meas_bearing = true_bearing + np.random.normal(0, sigma_bearing)
        # Normalize bearing between -pi and pi
        meas_bearing = (meas_bearing + np.pi) % (2 * np.pi) - np.pi
        
        # Target SNR (stronger than typical clutter)
        target_snr = np.random.uniform(12, 22)
        
        # Save true target measurement
        records.append({
            'time_step': t,
            'is_target': 1,
            'true_x': x,
            'true_y': y,
            'true_vx': vx,
            'true_vy': vy,
            'true_omega': omega,
            'meas_range': meas_range,
            'meas_bearing': meas_bearing,
            'snr': target_snr
        })
        
        # Generate false alarms (clutter) for this time step
        num_clutter = np.random.poisson(clutter_rate)
        for _ in range(num_clutter):
            c_range = np.random.uniform(min_range, max_range)
            c_bearing = np.random.uniform(min_bearing, max_bearing)
            c_snr = np.random.uniform(5, 12) # Lower SNR for random clutter
            
            records.append({
                'time_step': t,
                'is_target': 0,
                'true_x': '',
                'true_y': '',
                'true_vx': '',
                'true_vy': '',
                'true_omega': '',
                'meas_range': c_range,
                'meas_bearing': c_bearing,
                'snr': c_snr
            })
            
    # Save to CSV
    if records:
        keys = records[0].keys()
        with open(output_csv, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(records)
    
    print(f"Simulation complete!")
    print(f"Generated {len(records)} total hits: {num_steps} target hits, {len(records)-num_steps} false alarms/clutter.")
    print(f"Data saved to: {os.path.abspath(output_csv)}")

if __name__ == '__main__':
    # Run the simulation with 150 seconds of data, dt=1s, avg 100 clutter hits per scan
    # Outputting directly in the same folder as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'sonar_simulation.csv')
    simulate_sonar_data(num_steps=150, dt=1.0, clutter_rate=100, output_csv=output_path)
