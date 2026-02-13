#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>

#define ZERO 1.E-10 
#define BIG 1.E10

template <typename scalar_t>
__device__ __forceinline__ scalar_t getsoilPercRate(scalar_t soilTheta,
                                                    scalar_t soilFieldCap,
                                                    scalar_t soilPorosity,
                                                    scalar_t soilKS,
                                                    scalar_t soilKSlope)
{
//   scalar_t delta;
//   // printf("soilTheta:%f\t soilFieldCap:%f\n", soilTheta, soilFieldCap);
//   if (soilTheta <= soilFieldCap)
//   {
//     return 0.0;
//   }
//   delta = soilPorosity - soilTheta;
//   // printf("soilTheta:%f\t soilFieldCap:%f\t result:%f\n", soilTheta, soilFieldCap, soilKS * exp(-delta * soilKSlope));
//   return soilKS * exp(-delta * soilKSlope);

// NEW POWER FUNCTION - De-Ville et al. (2025) Equation (3)
//   K = Ks × [(θ - θfc)/(θs - θfc)]^2.5
    if (soilTheta <= soilFieldCap)
    {
        return 0.0;
    }
    // Calculate effective saturation: Sact = (θ - θfc) / (θs - θfc)
    scalar_t Sact = (soilTheta - soilFieldCap) / (soilPorosity - soilFieldCap);
    // Ensure Sact is in valid range [0, 1] for numerical stability
    Sact = max(static_cast<scalar_t>(0.0), 
                min(static_cast<scalar_t>(1.0), Sact));
    // Power function with n = 2.5 (from paper's calibration)
    scalar_t n = soilKSlope;
    scalar_t percRate = soilKS * pow(Sact, n);
    return percRate;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
soilLayer(scalar_t theta, scalar_t soilThickness, scalar_t soilPorosity,
          scalar_t soilFieldCap, scalar_t soilWiltPoint, scalar_t soilKS,
          scalar_t soilKSlope, scalar_t &maxRate, scalar_t tstep)
{
  scalar_t soilTheta = theta;

  scalar_t soilPerc = getsoilPercRate(soilTheta, soilFieldCap, soilPorosity,
                                      soilKS, soilKSlope);
  // printf("soilTheta:%f\t soilFieldCap:%f\n", soilTheta, soilFieldCap);
  scalar_t availVolume = (soilTheta - soilFieldCap) * soilThickness;
  maxRate = max(availVolume, 0.0) / tstep;

  soilPerc = min(soilPerc, maxRate);
  soilPerc = max(soilPerc, 0.0);
  return soilPerc;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t getStorageDrainRate(
    scalar_t storageDepth, scalar_t theta, scalar_t surfaceDepth,
    scalar_t paveDepth, scalar_t soilThickness, scalar_t soilPorosity,
    scalar_t soilFieldCap, scalar_t storageThickness, scalar_t paveThickness,
    scalar_t drainCoeff, scalar_t drainOffset, scalar_t drainExpon,
    const int lidparaIndex)
{
  scalar_t head = storageDepth;
  scalar_t outflow = 0.0;

  if (storageDepth >= storageThickness)
  {
    if (soilThickness > 0.0)
    {
      if (theta > soilFieldCap)
      {
        head += (theta - soilFieldCap) / (soilPorosity - soilFieldCap) *
                soilThickness;
        if (theta >= soilPorosity)
        {
          if (paveThickness > 0.0)
          {
            head += paveDepth;
          }
          else
          {
            head += surfaceDepth;
          }
        }
      }
    }
    if (paveThickness > 0.0)
    {
      head += paveDepth;
      if (paveDepth >= paveThickness)
      {
        head += surfaceDepth;
      }
    }
  }
  head -= drainOffset;
  if (head > ZERO)
  {
    outflow = drainCoeff * pow(head, drainExpon);
    static int count = 0;
    if (count < 3) {
        printf("Drain: coeff=%.6f head=%.6f exp=%.2f -> out=%.6e\n",
               drainCoeff, head, drainExpon, outflow);
        count++;
    }
  }
  return outflow;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t getDrainMatOutflow(
    scalar_t storageDepth, scalar_t SoilPerc, scalar_t drainMatAlpha,
    scalar_t dx, scalar_t dy, scalar_t drainMatvoidFrac)
{
  // scalar_t result = SoilPerc;
  scalar_t result = 0;
  if (drainMatAlpha > 0.0)
  {
    result = drainMatAlpha * pow(storageDepth, 5.0 / 3.0) *
             dy / (dx * dy) * drainMatvoidFrac;
  }
  return result;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t getPavementPermRate(scalar_t clogFactor, scalar_t hTreated, scalar_t kSat)
//
//  Purpose: computes reduced permeability of a pavement layer due to
//           clogging.
//  Input:   none
//  Output:  returns the reduced permeability of the pavement layer (ft/s).
//
{
  double permReduction = 0.0;
  if (clogFactor > 0.0)
  {
    // ... find permeabiity reduction factor
    permReduction = hTreated / clogFactor;
    permReduction = min(permReduction, 1.0);
  }
  // ... return the effective pavement permeability
  return kSat * (1.0 - permReduction);
}

template <typename scalar_t>
__device__ __forceinline__ void modpls_solve(scalar_t *x, scalar_t *f,
                                             scalar_t *xMin, scalar_t *xMax,
                                             scalar_t tstep)
{
  int i;
  scalar_t xOld[4];

  for (i = 0; i < 4; i++)
  {
    xOld[i] = x[i];
    if (i > 0)
    {
      x[i] = xOld[i] + f[i] * tstep;
      // printf("i:%d\t xold:%f\t f:%f\t x:%f\t xMax:%f\t areaRatio:%f\n", i,xOld[i],f[i],x[i],xMax[i],areaRatio);
    }
    else
    {
      x[i] = f[i] * tstep;
      // printf("i:%d\t xold:%f\t f:%f\t x:%f\t xMax:%f\t areaRatio:%f\n",i,xOld[0],f[0],x[0],xMax[0],areaRatio);
    }
    x[i] = min(x[i], xMax[i]);
    x[i] = max(x[i], xMin[i]);
  }
}

template <typename scalar_t>
__global__ void lidCalculation_kernel(
    int N, int32_t *__restrict__ wetMask_sorted, scalar_t *__restrict__ h_update,
    scalar_t *__restrict__ h, scalar_t *__restrict__ df, 
    uint8_t *__restrict__ landuseMask, uint8_t *__restrict__ lidMask,
    int n_landuse, int num_lid_types, scalar_t *__restrict__ areaMask,
    scalar_t *__restrict__ SurPara, scalar_t *__restrict__ SoilPara,
    scalar_t *__restrict__ StorPara, scalar_t *__restrict__ PavePara,
    scalar_t *__restrict__ DrainPara, scalar_t *__restrict__ DraMatPara,
    scalar_t *__restrict__ soilLimMin, scalar_t *__restrict__ soilLimMax,
    scalar_t *__restrict__ paveLimMax, scalar_t *__restrict__ storLimMax,
    scalar_t *__restrict__ cumuSurfaceWaterDepth,
    scalar_t *__restrict__ cumuSoilMoisture,
    scalar_t *__restrict__ cumuStorageWaterDepth,
    scalar_t *__restrict__ cumuPavementWaterDepth, 
    scalar_t *__restrict__ drainrate,
    scalar_t *__restrict__ dx, scalar_t *__restrict__ dy, 
    scalar_t *__restrict__ dt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N)
    // LID Type definitions:
    // 1 = Bio-Retention Cell (Biocell)
    // 2 = Rain Garden
    // 3 = Green Roof
    // 4 = Infiltration Trench
    // 5 = Permeable Pavement
    // 6 = Vegetative Swale (no calculation)
    // 7 = Rain Barrel (no calculation)
    {
        int32_t i = wetMask_sorted[j];
        uint8_t land_type = landuseMask[i];
        uint8_t lid_type = lidMask[i];
        scalar_t areaRatio = areaMask[i];

        // early exit if no LID
        if(lid_type < 1 || lid_type == 7) return;

        scalar_t tstep = dt[0];
        uint8_t lidparaIndex = land_type - n_landuse;

        // load parameters
        scalar_t surfaceThickness = SurPara[lidparaIndex + 0 * num_lid_types];
        scalar_t surfaceVoidFrac = SurPara[lidparaIndex + 1 * num_lid_types];
        scalar_t surfaceRough = SurPara[lidparaIndex + 2 * num_lid_types];

        scalar_t soilThickness = SoilPara[lidparaIndex + 0 * num_lid_types];
        scalar_t soilPorosity = SoilPara[lidparaIndex + 1 * num_lid_types];
        scalar_t soilFieldCap = SoilPara[lidparaIndex + 2 * num_lid_types];
        scalar_t soilWiltPoint = SoilPara[lidparaIndex + 3 * num_lid_types];
        scalar_t soilKS = SoilPara[lidparaIndex + 4 * num_lid_types];
        scalar_t soilKSlope = SoilPara[lidparaIndex + 5 * num_lid_types];

        scalar_t storageThickness = StorPara[lidparaIndex + 0 * num_lid_types];
        scalar_t storageVoidFrac = StorPara[lidparaIndex + 1 * num_lid_types];
        scalar_t storageKS = StorPara[lidparaIndex + 2 * num_lid_types];
        scalar_t storageClog = StorPara[lidparaIndex + 3 * num_lid_types];

        scalar_t paveThickness = PavePara[lidparaIndex + 0 * num_lid_types];
        scalar_t paveVoidFrac = PavePara[lidparaIndex + 1 * num_lid_types];
        scalar_t paveImpervFrac = PavePara[lidparaIndex + 2 * num_lid_types];
        scalar_t paveKSat = PavePara[lidparaIndex + 3 * num_lid_types];
        scalar_t paveClogFactor = PavePara[lidparaIndex + 4 * num_lid_types];

        scalar_t drainCoeff = DrainPara[lidparaIndex + 0 * num_lid_types];
        scalar_t drainExpon = DrainPara[lidparaIndex + 1 * num_lid_types];
        scalar_t drainOffset = DrainPara[lidparaIndex + 2 * num_lid_types];

        scalar_t drainMatThickness = DraMatPara[lidparaIndex + 0 * num_lid_types];
        scalar_t drainMatVoidFrac = DraMatPara[lidparaIndex + 1 * num_lid_types];
        scalar_t drainMatAlpha = DraMatPara[lidparaIndex + 2 * num_lid_types];

        // Adjust pavement parameters
        scalar_t paveFraction = 1.0 - paveImpervFrac;

        // Initialize State Variables
        scalar_t x[4];  // [surface, soil_theta, storage, pavement]
        scalar_t f[4];  // Flux rates
        scalar_t xMin[4], xMax[4];

        // Load previous timestep states
        scalar_t soilTheta = cumuSoilMoisture[i];
        scalar_t storageDepth = cumuStorageWaterDepth[i];
        scalar_t paveDepth = cumuPavementWaterDepth[i];

        x[0] = 0.0;  // Will be set differently for each type
        x[1] = soilTheta;
        x[2] = storageDepth;
        x[3] = paveDepth;

        // Set bounds
        xMin[0] = 0.0;
        xMin[1] = soilWiltPoint;
        xMin[2] = 0.0;
        xMin[3] = 0.0;

        xMax[0] = BIG;  // Will be adjusted per type
        xMax[1] = soilPorosity;
        xMax[2] = storageThickness * storageVoidFrac;
        xMax[3] = paveThickness * paveVoidFrac;

        // Calculate Fluxes Based on LID Type
        scalar_t SurfaceInfil = 0.0;
        scalar_t SoilPerc = 0.0;
        scalar_t StorageDrain = 0.0;
        scalar_t PavePerc = 0.0;
        scalar_t StorageExfil = 0.0;
        scalar_t maxRate = 0.0;
        scalar_t availVolume = 0.0;
        scalar_t storageInflow = 0.0;

        // TYPE 3: GREEN ROOF
        if (lid_type == 3) // Green Roof
        {
            // Use drainage mat as storage layer
            storageThickness = drainMatThickness;
            storageVoidFrac = drainMatVoidFrac;
            xMax[2] = storageThickness * storageVoidFrac;

            // Surface to Soil Infiltration
            SurfaceInfil = (x[0] / tstep) * surfaceVoidFrac;
            f[0] = -SurfaceInfil;

            // Surface infiltration from Kernel 1
            SurfaceInfil = df[i] / tstep;

            // Soil percolation
            SoilPerc = soilLayer(soilTheta, soilThickness, soilPorosity, 
                           soilFieldCap, soilWiltPoint, soilKS, 
                           soilKSlope, maxRate, tstep);
            
            availVolume = (soilTheta - soilFieldCap) * soilThickness;
            maxRate = max(availVolume, 0.0) / tstep;
            SoilPerc = min(SoilPerc, maxRate);
            SoilPerc = max(SoilPerc, 0.0);

            // Drainage mat outflow
            StorageDrain = getDrainMatOutflow(storageDepth, SoilPerc, drainMatAlpha,
                                        dx[0], dy[0], storageVoidFrac);
            
            // Apply flux limiters
            if (soilTheta >= soilPorosity && storageDepth >= storageThickness) //unit is full
            {
                // both layers full
                maxRate = min(SoilPerc, StorageDrain);
                SoilPerc = maxRate;
                StorageDrain = maxRate;
                // adjust inflow rate to soil layer
                SurfaceInfil = min(SurfaceInfil, maxRate);
            }
            else{
                // limit drainage by available water
                maxRate = storageDepth * storageVoidFrac / tstep;
                if (storageDepth >= storageThickness)
                {
                    maxRate += SoilPerc;
                }
                maxRate = max(maxRate, 0.0);
                StorageDrain = min(StorageDrain, maxRate);

                // limit soil perc inflow by unused storage volume
                maxRate = (storageThickness - storageDepth) * storageVoidFrac / tstep +
                  StorageDrain;
                SoilPerc = min(SoilPerc, maxRate);

                // adjust surface infil. so soil porosity not exceeded
                maxRate = (storageThickness - storageDepth) * storageVoidFrac / tstep 
                  + StorageDrain;
                SurfaceInfil = min(SurfaceInfil, maxRate);
            }

            f[0] = -SurfaceInfil;
            f[1] = (SurfaceInfil - SoilPerc) / soilThickness;
            f[2] = (SoilPerc - StorageDrain) / storageVoidFrac;
            f[3] = 0.0;

            x[0] = cumuSurfaceWaterDepth[i];
            xMax[0] = surfaceThickness;
            drainrate[i] = StorageDrain;            
        }
        else if (lid_type == 1 || lid_type == 2)
        {
            // initial guess from Knernel 1
            SurfaceInfil = df[i] / tstep;
            // soil percolation
            SoilPerc = soilLayer(soilTheta, soilThickness, soilPorosity, 
                           soilFieldCap, soilWiltPoint, soilKS, 
                           soilKSlope, maxRate, tstep);
            //  assume it is imperviouse bottom
            if (drainCoeff > 0.0)
            {
                StorageDrain = getStorageDrainRate(
                    storageDepth, soilTheta, x[0], paveDepth, soilThickness,
                    soilPorosity, soilFieldCap, storageThickness, paveThickness,
                    drainCoeff, drainOffset, drainExpon, lidparaIndex);
            }

            if (storageThickness == 0.0) // scenario 1: there is no storage layer (rain garden)
            {
                SoilPerc = 0.0; // assume soil layer is the bottom layer
                // storaeExfil = maxRate;
                maxRate = (soilPorosity - soilTheta) * soilThickness / tstep + SoilPerc; // max infil rate cannot exceed soil porosity
                SurfaceInfil = min(SurfaceInfil, maxRate);
            }
            else if (soilTheta >= soilPorosity && storageDepth >= storageThickness) // soil & storage layers are full
            {
                maxRate = StorageExfil + StorageDrain; // the system is full, max limit by outflow rate
                if (SoilPerc < maxRate) // soil perc is the limiting rate
                {
                    maxRate = SoilPerc;
                    if (maxRate > StorageExfil)
                    {
                        StorageDrain = maxRate - StorageExfil; // exfill first and then drain
                    }
                    else
                    {
                        StorageExfil = maxRate; // all exfile, no drain as the soil perc is smalller than exfil
                        StorageDrain = 0.0;
                    }
                }
                else // soil perc is larger than outflow rate
                {
                    SoilPerc = maxRate; // limit soil perc to outflow rate
                }
                SurfaceInfil = min(SurfaceInfil, maxRate);
            }
            else if( storageThickness > 0.0) // normal scenario with soil and storage layers
            {
                StorageExfil = 0.0;

                // maxRate = SoilPerc + storageDepth * storageVoidFrac / tstep;
                // printf("maxRate:%f\t SoilPerc:%f\t storageDepth:%f\t storageVoidFrac:%f\n",maxRate, SoilPerc, storageDepth, storageVoidFrac);
                // StorageExfil = min(StorageExfil, maxRate);
                // StorageExfil = max(StorageExfil, 0.0);
                // if (StorageDrain > 0.0)
                // {
                //     // Calculate available water for drainage
                //     // = water percolating in + water already in storage - water infiltrating out
                //     maxRate = SoilPerc;  // Water coming from soil
                    
                //     // if (drainOffset <= storageDepth)
                //     if (storageDepth > drainOffset + 1e-10)
                //     {
                //         // Add water available in storage above drain offset
                //         maxRate += (storageDepth - drainOffset) * storageVoidFrac / tstep;
                //     }
                    
                //     // Subtract water going out the bottom
                //     maxRate -= StorageExfil;
                    
                //     // Make sure it's not negative
                //     maxRate = max(maxRate, 0.0);
                //     StorageDrain = min(StorageDrain, maxRate);

                // }
                // // Limit soil perc by unused storage volume
                // maxRate = StorageExfil + StorageDrain +
                //         (storageThickness - storageDepth) * storageVoidFrac / tstep;
                // SoilPerc = min(SoilPerc, maxRate);
                
                // // Limit surface infil. by unused soil volume
                // maxRate = (soilPorosity - soilTheta) * soilThickness / tstep + SoilPerc;
                // SurfaceInfil = min(SurfaceInfil, maxRate);

                maxRate = SoilPerc + storageDepth * storageVoidFrac / tstep; 
                StorageExfil = min(StorageExfil, maxRate); // max infil rate cannot exceed the current storage depth + perc water from soil
                StorageExfil = max(StorageExfil, 0.0);
                if (StorageDrain > 0.0)
                {
                    // calculate current time step, available water for drainage
                    // = water percolating in + water already in storage above drain offset - water infiltrating out
                    scalar_t availableWater = SoilPerc; 
                    if (storageDepth > drainOffset + 1e-10)
                    {
                        availableWater += (storageDepth - drainOffset) * storageVoidFrac / tstep;
                    }
                    // first satisfy the exfiltration and then the drain
                    scalar_t maxDrain = max(availableWater - StorageExfil, 0.0);
                    // limit the drainage rate
                    StorageDrain = min(StorageDrain, maxDrain);
                }
                //... limit soil perc by unused storage volume //StorageExfil +
                maxRate = StorageExfil + StorageDrain +
                        (storageThickness - storageDepth) * storageVoidFrac / tstep;
                SoilPerc = min(SoilPerc, maxRate);
                //... limit surface infil. by unused soil volume
                maxRate = (soilPorosity - soilTheta) * soilThickness / tstep + SoilPerc;
                SurfaceInfil = min(SurfaceInfil, maxRate);
            }
            f[0] = -SurfaceInfil;
            f[1] = (SurfaceInfil - SoilPerc) / soilThickness;
            if (storageThickness == 0.0)
            {
                f[2] = 0.0;
            }
            else
            {
                f[2] = (SoilPerc - StorageExfil - StorageDrain) / storageVoidFrac;
            }
            f[3] = 0.0;
            drainrate[i] = StorageDrain;
            x[0] = 0.0;
            printf("[OUTFLOW_DEBUG] i=%d, Theta=%.6f, StorDepth=%.6f, Outflow=%.8f\n", 
                i, soilTheta, storageDepth, StorageDrain);

        }
        else if (lid_type == 4) // Infiltration Trench
        {
            SurfaceInfil = df[i] / tstep;
            if (drainCoeff > 0.0)
            {
                StorageDrain = getStorageDrainRate(
                    storageDepth, soilTheta, x[0], paveDepth, soilThickness,
                    soilPorosity, soilFieldCap, storageThickness, paveThickness,
                    drainCoeff, drainOffset, drainExpon, lidparaIndex);
            }
            // limit underdrain flow by volume above drain offset
            if (StorageDrain > 0.0)
            {
                maxRate = 0;
                if (storageDepth >= storageThickness)
                {
                    maxRate += storageInflow;
                }
                    
                if (drainOffset <= storageDepth)
                {
                maxRate += (storageDepth - drainCoeff) *
                            storageVoidFrac / tstep;
                }
                maxRate = max(maxRate, 0.0);
                StorageDrain = min(StorageDrain, maxRate);
            }
            maxRate = (storageThickness - storageDepth) * storageVoidFrac / tstep +
                StorageDrain;
            storageInflow = min(storageInflow, maxRate);
            SurfaceInfil = storageInflow;

            f[0] = SurfaceInfil;
            f[1] = 0.0;
            f[2] = (SurfaceInfil - StorageDrain) / storageVoidFrac;
            f[3] = 0.0;
            x[0] = 0.0;
        }
        
        else if (lid_type == 5) // Permeable Pavement
        {
            // Surface to pavement infiltration
            scalar_t SurfaceInfil = df[i] / tstep;
            PavePerc = getPavementPermRate(paveClogFactor, h[i], paveKSat);
            SurfaceInfil = min(SurfaceInfil, PavePerc);
            //limit pavement perc by available water
            maxRate = (paveDepth * paveVoidFrac) / tstep + SurfaceInfil;
            maxRate = max(maxRate, 0.0);
            PavePerc = min(PavePerc, maxRate);

           if (soilThickness > 0.0)
            {
                SoilPerc = soilLayer(soilTheta, soilThickness, soilPorosity, soilFieldCap,
                                    soilWiltPoint, soilKS, soilKSlope, maxRate, tstep);
                // printf("========SoilPerc:%f\n",SoilPerc);
                availVolume = (soilTheta - soilFieldCap) * soilThickness;
                maxRate = max(availVolume, 0.0) / tstep;
                SoilPerc = min(SoilPerc, maxRate);
                SoilPerc = max(SoilPerc, 0.0);
                // printf("SoilPerc:%f\n",SoilPerc);
            }
            else
            {
                SoilPerc = PavePerc;
            }
            StorageDrain = 0.0;
            if (drainCoeff > 0.0)
            {
                StorageDrain = getStorageDrainRate(storageDepth, soilTheta, x[0], paveDepth, soilThickness,
                                                soilPorosity, soilFieldCap, storageThickness, paveThickness,
                                                drainCoeff, drainOffset, drainExpon, lidparaIndex);
            }
            //... no soil layer, pavement & storage layers are full
            if (soilThickness == 0.0 &&
                storageDepth >= storageThickness &&
                paveDepth >= paveThickness)
            {
                //... pavement outflow can't exceed storage outflow
                maxRate = StorageDrain;
                if (PavePerc > maxRate)
                {
                    PavePerc = maxRate;
                }
                //... storage outflow can't exceed pavement outflow
                else
                {
                    StorageDrain = PavePerc;
                }

                //... set soil perc to pavement perc
                SoilPerc = PavePerc;
                //... limit surface infil. by pavement perc
                // printf("SoilPerc:%f\n",SoilPerc);
                SurfaceInfil = min(SurfaceInfil, PavePerc);
            }
            else if (soilThickness > 0 &&
                    storageDepth >= storageThickness &&
                    soilTheta >= soilPorosity &&
                    paveDepth >= paveThickness)
            {
                //... find which layer has limiting flux rate
                maxRate = StorageDrain;
                if (SoilPerc < maxRate)
                {
                    maxRate = SoilPerc;
                }
                else
                {
                    maxRate = min(maxRate, PavePerc);
                }
                SoilPerc = maxRate;
                PavePerc = maxRate;
                //... limit surface infil. by pavement perc
                SurfaceInfil = min(SurfaceInfil, PavePerc);
                
            }
            //... storage & soil layers are full
            else if (soilThickness > 0.0 &&
                    storageDepth >= storageThickness &&
                    soilTheta >= soilPorosity)
            {
                //... soil perc can't exceed storage outflow
                maxRate = StorageDrain;
                if (SoilPerc > maxRate)
                { 
                    SoilPerc = maxRate;
                }

                //... storage outflow can't exceed soil perc
                else
                {
                    StorageDrain = SoilPerc;
                }

                //... limit surface infil. by available pavement volume
                availVolume = (paveThickness - paveDepth) * paveVoidFrac;
                maxRate = availVolume / tstep + PavePerc;
                SurfaceInfil = min(SurfaceInfil, maxRate);
            }
            //... soil and pavement layers are full
            else if (soilThickness > 0.0 &&
                    paveDepth >= paveThickness &&
                    soilTheta >= soilPorosity)
            {
                PavePerc = min(PavePerc, SoilPerc);
                SoilPerc = PavePerc;
                SurfaceInfil = min(SurfaceInfil, PavePerc);
            }
            else
            {
                maxRate = SoilPerc + (storageDepth * storageVoidFrac) / tstep;
                // printf("storageDepth:%f\t storageVoidFrac:%f\n",storageDepth, storageVoidFrac);
                maxRate = max(0.0, maxRate);
                // printf("maxRate:%f\n",maxRate);
                //... limit underdrain flow by volume above drain offset
                if (StorageDrain > 0.0)
                {
                    maxRate = 0;
                    if (storageDepth >= storageThickness)
                    {
                        maxRate += SoilPerc;
                    }
                        
                    if (drainOffset <= storageDepth)
                    {
                        maxRate += (storageDepth - drainOffset) *
                                storageVoidFrac / tstep;
                    }
                    maxRate = max(maxRate, 0.0);
                    StorageDrain = min(StorageDrain, maxRate);
                }
                //... limit soil & pavement outflow by unused storage volume
                availVolume = (storageThickness - storageDepth) * storageVoidFrac;
                maxRate = availVolume / tstep + StorageDrain;
                maxRate = max(maxRate, 0.0);
                if (soilThickness > 0.0)
                {
                    SoilPerc = min(SoilPerc, maxRate);
                    maxRate = (soilPorosity - soilTheta) * soilThickness / tstep +
                                SoilPerc;
                }
                PavePerc = min(PavePerc, maxRate);

                //... limit surface infil. by available pavement volume
                availVolume = (paveThickness - paveDepth) * paveVoidFrac;
                maxRate = availVolume / tstep + PavePerc;
                SurfaceInfil = min(SurfaceInfil, maxRate);
            }
            f[0] = SurfaceInfil;
            f[3] = (SurfaceInfil - PavePerc) / paveVoidFrac;
            if (soilThickness > 0.0)
            {
                f[1] = (PavePerc - SoilPerc) / soilThickness;
                storageInflow = SoilPerc;
            }
            else
            {
                f[1] = 0.0;
                storageInflow = PavePerc;
                SoilPerc = 0.0;
            }
            f[2] = (storageInflow - StorageDrain) / storageVoidFrac;
            drainrate[i] = StorageDrain;
            x[0] = 0.0;
        }
        
        modpls_solve(x, f, xMin, xMax, tstep);

        // update states
        if (lid_type == 3)
        {
            cumuSurfaceWaterDepth[i] = x[0];
            // Check for overflow
            if (cumuSurfaceWaterDepth[i] > surfaceThickness) {
                scalar_t excess = cumuSurfaceWaterDepth[i] - surfaceThickness;
                cumuSurfaceWaterDepth[i] = surfaceThickness;
                h_update[i] += excess;
            }
            cumuSoilMoisture[i] = x[1];
            cumuStorageWaterDepth[i] = x[2];
            cumuPavementWaterDepth[i] = 0.0;
        }
        else
        {
            // Calculate actual vs potential infiltration
            scalar_t actualInfil = -f[0] * tstep;
            scalar_t potentialInfil = df[i];
            
            // Refund over-deduction
            if (actualInfil < potentialInfil) 
            {
                scalar_t correction = potentialInfil - actualInfil;
                h_update[i] += correction;
            }
            cumuSoilMoisture[i] = x[1];
            cumuStorageWaterDepth[i] = x[2];
            cumuPavementWaterDepth[i] = x[3];
            cumuSurfaceWaterDepth[i] = 0.0;
        }
        // Safety bounds
        cumuSurfaceWaterDepth[i] = max(0.0, cumuSurfaceWaterDepth[i]);
        cumuSoilMoisture[i] = max(soilWiltPoint, min(soilPorosity, cumuSoilMoisture[i]));
        cumuStorageWaterDepth[i] = max(0.0, cumuStorageWaterDepth[i]);
        cumuPavementWaterDepth[i] = max(0.0, cumuPavementWaterDepth[i]);
    }

}

void lidCalculation_cuda(
    at::Tensor wetMask_sorted, at::Tensor h_update, at::Tensor landuseMask,
    at::Tensor lidMask,
    int n_landuse, at::Tensor areaMask, at::Tensor h, at::Tensor df,
    at::Tensor SurPara, at::Tensor SoilPara, at::Tensor StorPara,
    at::Tensor PavePara, at::Tensor DrainPara, at::Tensor DraMatPara,
    at::Tensor soilLimMin, at::Tensor soilLimMax, at::Tensor paveLimMax,
    at::Tensor storLimMax, at::Tensor cumuSurfaceWaterDepth,
    at::Tensor cumuSoilMoisture, at::Tensor cumuStorageWaterDepth,
    at::Tensor cumuPavementWaterDepth, at::Tensor drainrate, at::Tensor dx,
    at::Tensor dy, at::Tensor dt)
{
  const int N = wetMask_sorted.numel();

  if (N == 0)
  {
    return;
  }
  //  SoilPara 推断 LID 类型数量
  int num_lid_types = SoilPara.size(1);  // SoilPara shape: [7, n_lid_types]

  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int thread_0 = 256;
  int block_0 = (N + 256 - 1) / 256;
  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "lidCalculation_cuda", ([&]
                                        { lidCalculation_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
                                              N, wetMask_sorted.data<int32_t>(), h_update.data<scalar_t>(),
                                              h.data<scalar_t>(), df.data<scalar_t>(), landuseMask.data<uint8_t>(), lidMask.data<uint8_t>(),
                                              n_landuse, num_lid_types, areaMask.data<scalar_t>(),
                                              SurPara.data<scalar_t>(), SoilPara.data<scalar_t>(),
                                              StorPara.data<scalar_t>(), PavePara.data<scalar_t>(),
                                              DrainPara.data<scalar_t>(), DraMatPara.data<scalar_t>(),
                                              soilLimMin.data<scalar_t>(), soilLimMax.data<scalar_t>(),
                                              paveLimMax.data<scalar_t>(), storLimMax.data<scalar_t>(),
                                              cumuSurfaceWaterDepth.data<scalar_t>(),
                                              cumuSoilMoisture.data<scalar_t>(),
                                              cumuStorageWaterDepth.data<scalar_t>(),
                                              cumuPavementWaterDepth.data<scalar_t>(),
                                              drainrate.data<scalar_t>(),
                                              dx.data<scalar_t>(),
                                              dy.data<scalar_t>(), dt.data<scalar_t>()); }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Ini Error in load_textures: %s\n", cudaGetErrorString(err));
}
