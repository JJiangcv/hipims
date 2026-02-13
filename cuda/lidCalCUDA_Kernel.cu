#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>

#define ZERO 1.E-10 // Effective zero value
#define BIG 1.E10

template <typename scalar_t>
__device__ __forceinline__ scalar_t getsoilPercRate(scalar_t soilTheta,
                                                    scalar_t soilFieldCap,
                                                    scalar_t soilPorosity,
                                                    scalar_t soilKS,
                                                    scalar_t soilKSlope)
{
  scalar_t delta;
  // printf("soilTheta:%f\t soilFieldCap:%f\n", soilTheta, soilFieldCap);
  if (soilTheta <= soilFieldCap)
  {
    return 0.0;
  }
  delta = soilPorosity - soilTheta;
  // printf("soilTheta:%f\t soilFieldCap:%f\t result:%f\n", soilTheta, soilFieldCap, soilKS * exp(-delta * soilKSlope));
  return soilKS * exp(-delta * soilKSlope);
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

// x[2], x[1], x[0], 0.0, SoilPara, StorPara, PavePara, DrainPara, lidparaIndex

template <typename scalar_t>
__device__ __forceinline__ scalar_t getStorageDrainRate(
    scalar_t storageDepth, scalar_t theta, scalar_t surfaceDepth,
    scalar_t paveDepth, scalar_t soilThickness, scalar_t soilPorosity,
    scalar_t soilFieldCap, scalar_t storageThickness, scalar_t paveThickness,
    scalar_t drainCoeff, scalar_t drainoffset, scalar_t drainexpon,
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
  head -= drainoffset;
  if (head > ZERO)
  {
    outflow = drainCoeff * pow(head, drainexpon);
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
    scalar_t *__restrict__ h, scalar_t *__restrict__ df, uint8_t *__restrict__ landuseMask, uint8_t *__restrict__ lidMask,
    int n_landuse, int num_lid_types, scalar_t *__restrict__ areaMask,
    scalar_t *__restrict__ SurPara, scalar_t *__restrict__ SoilPara,
    scalar_t *__restrict__ StorPara, scalar_t *__restrict__ PavePara,
    scalar_t *__restrict__ DrainPara, scalar_t *__restrict__ DraMatPara,
    scalar_t *__restrict__ soilLimMin, scalar_t *__restrict__ soilLimMax,
    scalar_t *__restrict__ paveLimMax, scalar_t *__restrict__ storLimMax,
    scalar_t *__restrict__ cumuSurfaceWaterDepth,
    scalar_t *__restrict__ cumuSoilMoisture,
    scalar_t *__restrict__ cumuStorageWaterDepth,
    scalar_t *__restrict__ cumuPavementWaterDepth, scalar_t *__restrict__ drainrate,
    scalar_t *__restrict__ dx, scalar_t *__restrict__ dy, scalar_t *__restrict__ dt)
{
  // get the index of cell
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N)
  {
    int32_t i = wetMask_sorted[j];
    uint8_t land_type = landuseMask[i];
    uint8_t lidtype = lidMask[i];
    scalar_t areaRatio = areaMask[i];
    // LID Type definitions:
    // 1 = Bio-Retention Cell (Biocell)
    // 2 = Rain Garden
    // 3 = Green Roof
    // 4 = Infiltration Trench
    // 5 = Permeable Pavement
    // 6 = Vegetative Swale (no calculation)
    // 7 = Rain Barrel (no calculation)
    if (lidtype < 1 || lidtype == 7)
    {
      return;
    }
    scalar_t tstep = dt[0];
    uint8_t lidparaIndex = land_type - n_landuse;
    // printf("Index:%d\t num:%d\t land_type:%d\n", Index,num,land_type);
    scalar_t soilThickness = SoilPara[lidparaIndex + 0 * num_lid_types];
    scalar_t soilPorosity = SoilPara[lidparaIndex + 1 * num_lid_types];
    scalar_t soilFieldCap = SoilPara[lidparaIndex + 2 * num_lid_types];

    scalar_t soilWiltPoint = SoilPara[lidparaIndex + 3 * num_lid_types];
    scalar_t soilKS = SoilPara[lidparaIndex + 4 * num_lid_types];
    scalar_t soilKSlope = SoilPara[lidparaIndex + 5 * num_lid_types];
    scalar_t storageThickness = StorPara[lidparaIndex + 0 * num_lid_types];
    scalar_t storageVoidFrac = StorPara[lidparaIndex + 1 * num_lid_types];
    scalar_t paveThickness = PavePara[lidparaIndex + 0 * num_lid_types];
    scalar_t drainCoeff = DrainPara[lidparaIndex + 0 * num_lid_types];
    scalar_t paveFraction = 1 - PavePara[lidparaIndex + 2 * num_lid_types];
    scalar_t paveVoidFrac = PavePara[lidparaIndex + 1 * num_lid_types] * paveFraction;
    scalar_t pavekSat = PavePara[lidparaIndex + 3 * num_lid_types];
    scalar_t paveclogFactor = PavePara[lidparaIndex + 4 * num_lid_types];
    scalar_t drainexpon = DrainPara[lidparaIndex + 1 * num_lid_types];
    scalar_t drainoffset = DrainPara[lidparaIndex + 2 * num_lid_types];
    scalar_t surfaceThickness = SurPara[lidparaIndex + 0 * num_lid_types];
    scalar_t surfacevoid = SurPara[lidparaIndex + 1 * num_lid_types];

    scalar_t drainMatThickness = DraMatPara[lidparaIndex + 0 * num_lid_types];
    scalar_t drainMatFraction = DraMatPara[lidparaIndex + 1 * num_lid_types];
    scalar_t drainMatRough = DraMatPara[lidparaIndex + 2 * num_lid_types];
    scalar_t drainMatAlpha = DraMatPara[lidparaIndex + 3 * num_lid_types];
    scalar_t StorageExfil = 0.0;

    scalar_t x[4];
    scalar_t f[4];
    scalar_t xMin[4];
    scalar_t xMax[4];
    scalar_t SoilPerc = 0.0;
    scalar_t StorageDrain = 0.0;
    scalar_t PavePerc = 0.0;

    // printf("Befor Cal:h[i]:%f\n",h[i]);
    x[0] = h[i];
    x[1] = cumuSoilMoisture[i];
    x[2] = cumuStorageWaterDepth[i];
    x[3] = cumuPavementWaterDepth[i];

    scalar_t surfaceDepth = x[0];
    scalar_t soilTheta = x[1];
    scalar_t storageDepth = x[2];
    scalar_t paveDepth = x[3];
    f[0] = 0.0;
    f[1] = 0.0;
    f[2] = 0.0;
    f[3] = 0.0;
    xMin[0] = 0.0;
    xMin[1] = soilLimMin[land_type];
    xMin[2] = 0.0;
    xMin[3] = 0.0;
    xMax[0] = BIG;
    xMax[1] = soilLimMax[land_type];
    xMax[2] = storLimMax[land_type];
    xMax[3] = paveLimMax[land_type];
    scalar_t maxRate = 0.0;

    if (lidtype >= 1 && lidtype <= 2) // bio-cell
    {
      // printf("h_update[i]:%f\t df[i]:%f\n", h_update[i],df[i]);
      scalar_t SurfaceInfil = df[i] / tstep;
      SoilPerc = soilLayer(soilTheta, soilThickness, soilPorosity, soilFieldCap,
                           soilWiltPoint, soilKS, soilKSlope, maxRate, tstep);
      // printf("soilTheta:%f\t SoilPerc:%f\n", soilTheta,SoilPerc);
      // assume it is imperviouse bottom
      if (drainCoeff > 0.0)
      {
        StorageDrain = getStorageDrainRate(
            storageDepth, soilTheta, surfaceDepth, paveDepth, soilThickness,
            soilPorosity, soilFieldCap, storageThickness, paveThickness,
            drainCoeff, drainoffset, drainexpon, lidparaIndex);
      }

      if (storageThickness == 0.0)
      {
        maxRate = min(SoilPerc, 0.0); // storageexfil
        SoilPerc = maxRate;
        // storaeExfil = maxRate;
        maxRate = (soilPorosity - soilTheta) * soilThickness / tstep + SoilPerc;
        SurfaceInfil = min(SurfaceInfil, maxRate);
        // printf("SurfaceInfil:%f\t hupdate:%f\t dt:%f\n", SurfaceInfil,-h_update[i],tstep);
      }
      else if (soilTheta >= soilPorosity &&
               storageDepth >= storageThickness)
      {
        maxRate = StorageExfil + StorageDrain; //
        if (SoilPerc < maxRate)
        {
          maxRate = SoilPerc;
          if (maxRate > StorageExfil) // StorageExfil
          {
            StorageDrain = maxRate - StorageExfil; //
          }
          else
          {
            StorageExfil = maxRate;
            StorageDrain = 0.0;
          }
        }
        else
        {
          SoilPerc = maxRate;
        }
        SurfaceInfil = min(SurfaceInfil, maxRate);
      }
      else if (storageThickness > 0.0)
      {
        maxRate = SoilPerc + storageDepth * storageVoidFrac / tstep;
        StorageExfil = min(StorageExfil, maxRate);
        StorageExfil = max(StorageExfil, 0.0);
        if (StorageDrain > 0.0)
        {
          maxRate = -StorageExfil; // - StorageEvap;
          if (storageDepth >= storageThickness)
          {
            maxRate += SoilPerc;
          }
          if (drainoffset <= storageDepth)
          {
            maxRate += (storageDepth - drainoffset) *
                       storageVoidFrac / tstep;
          }
          maxRate = max(maxRate, 0.0);
          StorageDrain = min(StorageDrain, maxRate);
        }
        //... limit soil perc by unused storage volume //StorageExfil +
        maxRate = StorageExfil + StorageDrain +
                  (storageThickness - storageDepth) * storageVoidFrac / tstep;
        SoilPerc = min(SoilPerc, maxRate);
        //... limit surface infil. by unused soil volume
        maxRate = (soilPorosity - soilTheta) * soilThickness / tstep + SoilPerc;
        SurfaceInfil = min(SurfaceInfil, maxRate);
      }

      f[0] = SurfaceInfil; /// /surfacevoid SurPara[1, lidparaIndex]soilThickness:%f
      f[1] = (SurfaceInfil - SoilPerc) / soilThickness;
      // printf("SurfaceInfil:%f\t soilThickness:%f\t SoilPerc:%f\t dt:%f\n",SurfaceInfil, soilThickness, SoilPerc,tstep);
      if (storageThickness == 0.0)
      {
        f[2] = 0.0;
      }
      else
      {
        f[2] = (SoilPerc - StorageExfil - StorageDrain) / storageVoidFrac;
      }
      drainrate[i] = StorageDrain;
    }

    if (lidtype == 3) // Green roof
    {
      storageThickness = DraMatPara[lidparaIndex + 0 * num_lid_types];
      storageVoidFrac = DraMatPara[lidparaIndex + 1 * num_lid_types];
      // 🔍 只打印第一个遇到的Green Roof cell
      if (j == 0) {  // j是wetMask数组的索引，第一个wet cell
        printf("\n=== GREEN ROOF [i=%d, j=%d] ===\n", i, j);
        printf("BEFORE:\n");
        printf("  x[2]: %.6f\n", x[2]);
        printf("  soilTheta: %.6f, FC: %.6f\n", x[1], soilFieldCap);
        printf("  storThick: %.6f, storVoid: %.6f\n", storageThickness, storageVoidFrac);
      }
      scalar_t SurfaceInfil = df[i] / tstep;
      scalar_t availVolume;

      //... soil layer perc rate * surfacevoid
      SoilPerc = soilLayer(soilTheta, soilThickness, soilPorosity, soilFieldCap,
                           soilWiltPoint, soilKS, soilKSlope, maxRate, tstep);
      // printf("SoilPerc:%f\t",SoilPerc);

      //... limit perc rate by available water
      availVolume = (soilTheta - soilFieldCap) * soilThickness;
      maxRate = max(availVolume, 0.0) / tstep;
      SoilPerc = min(SoilPerc, maxRate);
      SoilPerc = max(SoilPerc, 0.0);

      //... storage (drain mat) outflow rate
      StorageDrain = getDrainMatOutflow(storageDepth, SoilPerc, drainMatAlpha,
                                        dx[0], dy[0], storageVoidFrac);

      //... unit is full if(soilTheta>=soilPorosity && storageDepth > storageThickness)
      if (soilTheta >= soilPorosity && storageDepth >= storageThickness)
      {
        //... outflow from both layers equals limiting rate
        maxRate = min(SoilPerc, StorageDrain);
        SoilPerc = maxRate;
        StorageDrain = maxRate;
        //... adjust inflow rate to soil layer
        SurfaceInfil = min(SurfaceInfil, maxRate);
      }
      //... unit not full
      else
      {
        //... limit drainmat outflow by available storage volume
        maxRate = storageDepth * storageVoidFrac / tstep;
        if (storageDepth >= storageThickness)
        {
          maxRate += SoilPerc;
        }
        maxRate = max(maxRate, 0.0);
        StorageDrain = min(StorageDrain, maxRate);

        //... limit soil perc inflow by unused storage volume
        maxRate = (storageThickness - storageDepth) * storageVoidFrac / tstep +
                  StorageDrain;
        // printf("StorageDrain:%f\t storageDepth:%f\t maxRate:%f\n",StorageDrain, storageDepth, maxRate);
        SoilPerc = min(SoilPerc, maxRate);

        //... adjust surface infil. so soil porosity not exceeded
        maxRate = (soilPorosity - soilTheta) * soilThickness / tstep +
                  SoilPerc;
        // printf("SurfaceInfil:%f\t SoilPerc:%f\t maxRate:%f\n",SurfaceInfil, SoilPerc, maxRate);
        SurfaceInfil = min(SurfaceInfil, maxRate);
      }
      // printf("surfaceinfil:%f\n",SurfaceInfil);
      f[0] = SurfaceInfil;
      f[1] = (SurfaceInfil - SoilPerc) / soilThickness;
      // printf("1: StorageDrain:%f\t SoilPerc:%f\n",StorageDrain, SoilPerc);
      f[2] = (SoilPerc - StorageDrain) / storageVoidFrac;
      if (j == 0) {
        printf("FLUX:\n");
        printf("  SoilPerc: %.8f, StorDrain: %.8f\n", SoilPerc, StorageDrain);
        printf("  f[2]: %.8f\n", f[2]);
        printf("  xMax[2]: %.6f\n", xMax[2]);
      }

      // printf("2: SoilPerc:%f\t StorageDrain:%f\t storageVoidFrac:%f\n",SoilPerc, StorageDrain, storageVoidFrac);
      drainrate[i] = StorageDrain;
    }

    if (lidtype == 4) // infiltration trent
    {
      scalar_t availVolume;
      scalar_t StorageInflow;
      scalar_t SurfaceInfil = df[i] / tstep;
      StorageInflow = SurfaceInfil;
      // printf("SurfaceInfil:%f\n", SurfaceInfil);
      //... underdrain flow rate
      StorageDrain = 0.0;
      if (drainCoeff > 0.0)
      {
        StorageDrain = getStorageDrainRate(
            storageDepth, soilTheta, surfaceDepth, paveDepth, soilThickness,
            soilPorosity, soilFieldCap, storageThickness, paveThickness,
            drainCoeff, drainoffset, drainexpon, lidparaIndex);
      }
      //... limit underdrain flow by volume above drain offset
      if (StorageDrain > 0.0)
      {
        maxRate = 0;
        if (storageDepth >= storageThickness)
          maxRate += StorageInflow;
        if (drainCoeff <= storageDepth)
        {
          maxRate += (storageDepth - drainCoeff) *
                     storageVoidFrac / tstep;
        }
        maxRate = max(maxRate, 0.0);
        StorageDrain = min(StorageDrain, maxRate);
      }
      //... limit storage inflow to not exceed storage layer capacity
      maxRate = (storageThickness - storageDepth) * storageVoidFrac / tstep +
                StorageDrain;
      // printf("storageThickness:%f\t storageDepth:%f\n storageVoidFrac:%f\n", storageThickness, storageDepth,storageVoidFrac);
      StorageInflow = min(StorageInflow, maxRate);
      // printf("StorageInflow:%f\n", StorageInflow);
      //... equate surface infil to storage inflow
      SurfaceInfil = StorageInflow;

      f[0] = SurfaceInfil;
      f[2] = (StorageInflow - StorageDrain) / storageVoidFrac;
      f[1] = 0.0;
      drainrate[i] = StorageDrain;
    }

    if (lidtype == 5) // permeable
    {
      scalar_t storageInflow;
      scalar_t availVolume;
      scalar_t SurfaceInfil = df[i] / tstep;
      //... find perc rate out of pavement layer
      PavePerc = getPavementPermRate(paveclogFactor, h[i], pavekSat);
      // printf("pavekSat：%ft PavePerc:%f\t SurfaceInfil:%f\n",pavekSat, PavePerc,SurfaceInfil);
      SurfaceInfil = min(SurfaceInfil, PavePerc);
      //... limit pavement perc by available water
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
        StorageDrain = getStorageDrainRate(storageDepth, soilTheta, surfaceDepth, paveDepth, soilThickness,
                                           soilPorosity, soilFieldCap, storageThickness, paveThickness,
                                           drainCoeff, drainoffset, drainexpon, lidparaIndex);
      }
      //... check for adjacent saturated layers

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
          maxRate = SoilPerc;
        else
          maxRate = min(maxRate, PavePerc);
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
          SoilPerc = maxRate;

        //... storage outflow can't exceed soil perc
        else
          StorageDrain = SoilPerc;

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
            maxRate += SoilPerc;
          if (drainoffset <= storageDepth)
          {
            maxRate += (storageDepth - drainoffset) *
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
      // printf("PavePerc:%f\t storageInflow:%f\t SoilPerc:%f\t SurfaceInfil:%f\n",PavePerc, storageInflow, SoilPerc,SurfaceInfil);
      drainrate[i] = StorageDrain;
    }

    // update
    modpls_solve(x, f, xMin, xMax, tstep);
    if (j == 0 && lidtype == 3) {
      printf("AFTER modpls_solve:\n");
      printf("  x[2]: %.6f\n", x[2]);
      printf("  delta: %.8f = %.8f * %.4f * %.4f\n", 
            f[2] * tstep * areaRatio, f[2], tstep, areaRatio);
    }
    // if (lidtype == 3 && (i % 500000 == 0 || j % 1000 == 0))  // 第一个wet cell
    // {
    //   printf("GR[%d]: x[1](soil)=%.6f x[2](storage)=%.6f f[1]=%.8f f[2]=%.8f areaRatio=%.4f\n",
    //         i, x[1], x[2], f[1], f[2], areaRatio);
    // }

    // h_update[i] = -x[0];
    // printf("x[0]:%f\t sv:%f\t h_update[i]:%f\n", x[0], surfacevoid, h_update[i]);
    // printf("h_update:%f\n", h_update[i]);
    // h_update[i] = (h_update[i] - x[0])/surfacevoid;
    // printf("h_update:%f\t h:%f\n", h_update[i], h[i]);

    if (lidtype == 3) // Green roof - FIXED
    {
      // 计算本时间步的净入流
      scalar_t netInflow = (h_update[i] - x[0]) + h[i];
      
      // 更新累积储水量
      scalar_t newSurfaceWater = cumuSurfaceWaterDepth[i] + netInflow;
      
      if (newSurfaceWater <= surfaceThickness)
      {
        // Berm未满：全部储存
        cumuSurfaceWaterDepth[i] = newSurfaceWater;
        h_update[i] = -h[i];  // 清空表面水
      }
      else
      {
        // Berm满了：超出部分成为径流
        cumuSurfaceWaterDepth[i] = surfaceThickness;
        h_update[i] = (newSurfaceWater - surfaceThickness) - h[i];
      }
      
      // 边界检查
      cumuSurfaceWaterDepth[i] = max(0.0, min(cumuSurfaceWaterDepth[i], surfaceThickness));
    }
    else
    {
      h_update[i] = (h_update[i] - x[0]) / surfacevoid;
      cumuSurfaceWaterDepth[i] = h_update[i] + h[i];
      if (surfaceThickness != 0)
      {
        // printf("h_update[%d]:%f\n h[%d]:%f\n", i, h_update[i], i , h[i]);
        cumuSurfaceWaterDepth[i] = min(cumuSurfaceWaterDepth[i], surfaceThickness);
      }
    }
    cumuSoilMoisture[i] = x[1];
    cumuStorageWaterDepth[i] = x[2];
    cumuPavementWaterDepth[i] = x[3];
    if (j == 0 && lidtype == 3) {
      printf("FINAL: cumuStorage[i] = %.6f\n\n", cumuStorageWaterDepth[i]);
    }
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