"""
Automated Laser Characterization & Multivariate Calibration Pipeline
Target: Tunable DFB / PIC Laser Setpoint Prediction & Static Map Generation
Author: Ankit 
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import pyvisa

class LaserCalibrationArchitect:
    def __init__(self, visa_sourcemeter: str, visa_osa: str):
        """Initialize instrument resource manager and connection handles."""
        self.rm = pyvisa.ResourceManager()
        try:
            self.smu = self.rm.open_resource(visa_sourcemeter)
            self.osa = self.rm.open_resource(visa_osa)
            print("Instruments connected successfully.")
        except Exception as e:
            print(f"Simulation mode active (Hardware not found: {e})")
            self.smu, self.osa = None, None

    def acquire_characterization_matrix(self, current_steps: np.ndarray, temp_steps: np.ndarray) -> pd.DataFrame:
        """
        Automates multi-channel sweep across injection currents and temperatures 
        to capture optical power, central wavelength, and SMSR.
        """
        data = []
        print(f"Starting automated characterization sweep: {len(current_steps)} currents x {len(temp_steps)} temperatures")
        
        for T in temp_steps:
            if self.smu:
                # Example command structure for hardware control
                self.smu.write(f"CONF:TEMP {T}")
                time.sleep(0.2)  # Thermal stabilization delay
                
            for I in current_steps:
                if self.smu and self.osa:
                    self.smu.write(f"SOUR:CURR {I}")
                    time.sleep(0.05)
                    # Fetch live optical metrics from OSA
                    power = float(self.osa.query("MEAS:POW?"))
                    wavelength = float(self.osa.query("MEAS:WAV?"))
                else:
                    # Simulated synthetic response for validation framework
                    power = 0.5 * I - 0.001 * (T - 25)**2
                    wavelength = 1550.0 + 0.012 * I + 0.08 * T
                
                data.append({
                    'Set_Current_mA': I,
                    'Set_Temp_C': T,
                    'Measured_Power_mW': power,
                    'Measured_Wavelength_nm': wavelength
                }
                
        return pd.DataFrame(data)

    def train_multivariate_model(self, df: pd.DataFrame):
        """
        Builds a multivariate static map/regression pipeline for inverse setpoint prediction 
        (Mapping target Wavelength & Power -> Required Current & Temperature).
        """
        # Features: Target optical metrics -> Inputs: Required electrical setpoints
        X = df[['Measured_Wavelength_nm', 'Measured_Power_mW']].values
        y_current = df['Set_Current_mA'].values
        
        # Polynomial feature mapping for nonlinear laser behavior
        model_current = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1e-3))
        model_current.fit(X, y_current)
        
        print("Multivariate calibration mapping model trained successfully.")
        return model_current

if __name__ == "__main__":
    # Execution entry for validation pipeline
    currents = np.linspace(20, 100, 10)
    temperatures = np.linspace(20, 40, 5)
    
    cal_engine = LaserCalibrationArchitect("GPIB0::24::INSTR", "GPIB0::12::INSTR")
    raw_data = cal_engine.acquire_characterization_matrix(currents, temperatures)
    model = cal_engine.train_multivariate_model(raw_data)
