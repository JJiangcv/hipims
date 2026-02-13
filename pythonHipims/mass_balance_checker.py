#!/usr/bin/env python3
"""
Mass Balance Verification Script
=================================
Add this to your simulation to verify water balance at each timestep
"""

import torch
import numpy as np

class MassBalanceChecker:
    """Track and verify water balance during simulation"""
    
    def __init__(self, device, dx, n_cells):
        self.device = device
        self.dx = dx  # Cell size (m)
        self.cell_area = dx * dx  # m²
        self.n_cells = n_cells
        
        # Cumulative trackers
        self.total_rainfall = 0.0
        self.total_infiltration = 0.0
        self.total_drainage = 0.0
        self.total_ET = 0.0
        
        # Storage tracking
        self.initial_surface_storage = 0.0
        self.initial_soil_storage = 0.0
        self.initial_storage_layer = 0.0
        
        # Time series
        self.time_series = []
        self.balance_errors = []
        
    def initialize(self, h_internal, cumuSoilMoisture, cumuStorageWaterDepth,
                  SoilPara, StorPara, lidMask, areaMask):
        """Record initial storage"""
        # Surface water
        self.initial_surface_storage = torch.sum(h_internal * self.cell_area).item()
        
        # Soil moisture storage
        # For each LID cell: storage = θ * soil_thickness * area * areaRatio
        lid_cells = lidMask >= 1
        if lid_cells.any():
            # This is simplified - actual calculation needs proper indexing
            soil_storage = torch.sum(
                cumuSoilMoisture[lid_cells] * 
                SoilPara[0, 0] *  # soil thickness - needs proper indexing
                self.cell_area * 
                areaMask[lid_cells]
            ).item()
            self.initial_soil_storage = soil_storage
            
            # Storage layer
            storage_layer = torch.sum(
                cumuStorageWaterDepth[lid_cells] * 
                self.cell_area * 
                areaMask[lid_cells]
            ).item()
            self.initial_storage_layer = storage_layer
    
    def update(self, t, dt, rainfall_rate, h_internal, df, drainrate, 
               cumuSoilMoisture, cumuStorageWaterDepth, 
               SoilPara, StorPara, lidMask, areaMask):
        """
        Check mass balance at current timestep
        
        Parameters:
        -----------
        t : float
            Current simulation time (s)
        dt : float
            Timestep (s)
        rainfall_rate : float
            Rainfall rate (m/s)
        h_internal : tensor
            Current surface water depth (m)
        df : tensor
            Infiltration flux at this timestep (m³)
        drainrate : tensor
            Drainage rate (m/s)
        """
        
        # Calculate fluxes in this timestep (m³)
        rainfall_volume = rainfall_rate * dt * self.cell_area * self.n_cells
        
        # Infiltration
        infiltration_volume = torch.sum(df).item()  # df is already in m³
        
        # Drainage
        lid_cells = lidMask >= 1
        if lid_cells.any():
            drainage_volume = torch.sum(
                drainrate[lid_cells] * dt * self.cell_area * areaMask[lid_cells]
            ).item()
        else:
            drainage_volume = 0.0
        
        # Update cumulative
        self.total_rainfall += rainfall_volume
        self.total_infiltration += infiltration_volume
        self.total_drainage += drainage_volume
        
        # Current storage
        current_surface = torch.sum(h_internal * self.cell_area).item()
        
        # Simplified soil and storage calculation
        if lid_cells.any():
            current_soil = torch.sum(
                cumuSoilMoisture[lid_cells] * 
                SoilPara[0, 0] *  # Need proper indexing
                self.cell_area * 
                areaMask[lid_cells]
            ).item()
            
            current_storage_layer = torch.sum(
                cumuStorageWaterDepth[lid_cells] * 
                self.cell_area * 
                areaMask[lid_cells]
            ).item()
        else:
            current_soil = 0.0
            current_storage_layer = 0.0
        
        # Water balance equation (m³):
        # Current Storage = Initial Storage + Rainfall - Infiltration - Drainage - ET
        total_current_storage = current_surface + current_soil + current_storage_layer
        total_initial_storage = (self.initial_surface_storage + 
                                self.initial_soil_storage + 
                                self.initial_storage_layer)
        
        expected_storage = (total_initial_storage + 
                           self.total_rainfall - 
                           self.total_infiltration - 
                           self.total_drainage - 
                           self.total_ET)
        
        balance_error = total_current_storage - expected_storage
        relative_error = abs(balance_error) / max(self.total_rainfall, 1e-10) * 100
        
        # Store data
        self.time_series.append({
            't': t,
            'rainfall': self.total_rainfall,
            'infiltration': self.total_infiltration,
            'drainage': self.total_drainage,
            'surface_storage': current_surface,
            'soil_storage': current_soil,
            'storage_layer': current_storage_layer,
            'total_storage': total_current_storage,
            'expected_storage': expected_storage,
            'balance_error': balance_error,
            'relative_error': relative_error
        })
        
        self.balance_errors.append(balance_error)
        
        # Print warning if error is large
        if abs(relative_error) > 1.0:  # 1% threshold
            print(f"⚠️  WARNING at t={t:.1f}s: Mass balance error = {balance_error:.6f} m³ ({relative_error:.2f}%)")
            print(f"   Rainfall: {self.total_rainfall:.3f} m³")
            print(f"   Infiltration: {self.total_infiltration:.3f} m³")
            print(f"   Drainage: {self.total_drainage:.3f} m³")
            print(f"   Current storage: {total_current_storage:.3f} m³")
            print(f"   Expected storage: {expected_storage:.3f} m³")
    
    def get_summary(self):
        """Print summary statistics"""
        if not self.time_series:
            print("No data collected")
            return
            
        errors = np.array(self.balance_errors)
        
        print("\n" + "="*70)
        print("MASS BALANCE SUMMARY")
        print("="*70)
        print(f"\nCumulative Fluxes:")
        print(f"  Total Rainfall:      {self.total_rainfall:.6f} m³")
        print(f"  Total Infiltration:  {self.total_infiltration:.6f} m³")
        print(f"  Total Drainage:      {self.total_drainage:.6f} m³")
        print(f"  Total ET:            {self.total_ET:.6f} m³")
        
        final = self.time_series[-1]
        print(f"\nStorage:")
        print(f"  Initial total:       {self.initial_surface_storage + self.initial_soil_storage + self.initial_storage_layer:.6f} m³")
        print(f"  Final total:         {final['total_storage']:.6f} m³")
        print(f"  Expected final:      {final['expected_storage']:.6f} m³")
        
        print(f"\nBalance Errors:")
        print(f"  Mean error:          {np.mean(errors):.6f} m³")
        print(f"  Max error:           {np.max(np.abs(errors)):.6f} m³")
        print(f"  RMS error:           {np.sqrt(np.mean(errors**2)):.6f} m³")
        print(f"  Final error:         {final['balance_error']:.6f} m³")
        print(f"  Final relative:      {final['relative_error']:.4f} %")
        
        if abs(final['relative_error']) < 0.1:
            print("\n✅ PASSED: Mass balance error < 0.1%")
        elif abs(final['relative_error']) < 1.0:
            print("\n⚠️  WARNING: Mass balance error < 1% but > 0.1%")
        else:
            print("\n❌ FAILED: Mass balance error > 1%")
        print("="*70)
    
    def save_timeseries(self, filename):
        """Save time series to CSV"""
        import pandas as pd
        df = pd.DataFrame(self.time_series)
        df.to_csv(filename, index=False)
        print(f"\n💾 Time series saved to: {filename}")


# ============================================================================
# Integration Example
# ============================================================================

def example_integration():
    """
    Example of how to integrate mass balance checker into bio_vali.py
    """
    
    code_example = '''
# In bio_vali.py, after initialization:

# Create mass balance checker
from mass_balance_checker import MassBalanceChecker
mb_checker = MassBalanceChecker(device, paraDict['dx'], 
                               numerical._h_internal.numel())

# Initialize after setting up LID
mb_checker.initialize(
    numerical._h_internal,
    numerical._cumuSoilMoisture,
    numerical._cumuStorageWaterDepth,
    numerical._SoilPara,
    numerical._StorPara,
    numerical._lidMask,
    numerical._areaMask
)

# In the main simulation loop:
while numerical.t.item() < paraDict['EndTime']:
    # ... existing simulation code ...
    
    # Get current rainfall rate
    current_rainfall_rate = get_current_rainfall(numerical.t.item(), rainfallMatrix)
    
    # Check mass balance every N steps
    if n % 10 == 0:  # Check every 10 steps
        mb_checker.update(
            numerical.t.item(),
            numerical.dt.item(),
            current_rainfall_rate,
            numerical._h_internal,
            numerical._f_dt,  # or appropriate infiltration flux variable
            numerical._drainrate,
            numerical._cumuSoilMoisture,
            numerical._cumuStorageWaterDepth,
            numerical._SoilPara,
            numerical._StorPara,
            numerical._lidMask,
            numerical._areaMask
        )
    
    n += 1

# After simulation:
mb_checker.get_summary()
mb_checker.save_timeseries(paraDict['OUTPUT_PATH'] + '/mass_balance.csv')
'''
    
    print(code_example)


if __name__ == "__main__":
    print("Mass Balance Verification Script")
    print("="*70)
    print("\nThis script provides tools to verify water balance during simulation.")
    print("\nIntegration example:")
    example_integration()
