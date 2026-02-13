#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <math.h>
#include <stdio.h>

#define ZERO 1.E-10 // Effective zero value

template <typename scalar_t>
__device__ __forceinline__ double grnampt_getF2(scalar_t f1, scalar_t c1,
                                                scalar_t K_s, scalar_t ts)
{
  // printf("grnampt_getF2\n");
  scalar_t f2 = f1;
  scalar_t f2min = 0.0;
  scalar_t df2 = 0.0;
  scalar_t c2;
  // --- find min. infil. volume
  f2min = f1 + K_s * ts;

  // --- use min. infil. volume for 0 moisture deficit
  if (c1 == 0.0)
  {
    // printf("3\n");
    return f2min;
  }

  // --- use direct form of G-A equation for small time steps
  //     and c1/f1 < 100
  if (ts < 10.0 && f1 > 0.01 * c1)
  {
    // printf("1\n");
    f2 = f1 + K_s * (1.0 + c1 / f1) * ts;
    return max(f2, f2min);
  }
  // --- use Newton-Raphson method to solve integrated G-A equation
  //     (convergence limit reduced from that used in previous releases)
  c2 = c1 * log((f1 + c1)) - K_s * ts;
  for (int i = 1; i <= 20; i++)
  {
    // printf("2\n");
    df2 = (f2 - f1 - c1 * log(f2 + c1) + c2) / (1.0 - c1 / (f2 + c1));
    if (abs(df2) < 0.00001)
    {
      return max(f2, f2min);
    }
    f2 -= df2;
  }

  return f2min;
}

template <typename scalar_t>
__device__ __forceinline__ double
grnampt_getSatInfil(scalar_t *soilFu,
                    scalar_t *soilIMD, int *Sat, scalar_t _h, scalar_t K_s,
                    scalar_t Fumax, scalar_t total_head, int i,
                    scalar_t tstep)
{
  scalar_t ia = _h / tstep;
  scalar_t c1 = total_head * soilIMD[i];
  scalar_t F2 = grnampt_getF2(soilFu[i], c1, K_s, tstep); // rnampt_getF2(_h, c1, K_s, tstep)
  scalar_t dF = F2 - soilFu[i];
  // --- all available water infiltrates -- set saturated state to false

  if (dF > ia * tstep)
  {
    dF = ia * tstep;
    Sat[i] = 0;
  }
  // --- update total infiltration and upper zone moisture deficit
  soilFu[i] += dF;
  soilFu[i] = min(soilFu[i], Fumax);
  return dF;
}

template <typename scalar_t>
__device__ __forceinline__ double
grnampt_getUnsatInfil(scalar_t *soilFu,
                      scalar_t *soilIMD, int *Sat, scalar_t _h, scalar_t K_s,
                      scalar_t Fumax, scalar_t total_head, int i,
                      scalar_t tstep)
{
  scalar_t ia = _h / tstep;
  scalar_t dF = 0.0;

  if (ia < ZERO)
  {
    ia = 0;
  }
  if (ia <= K_s)
  {
    dF = ia * tstep;
    soilFu[i] += dF;
    return dF;
  }
  scalar_t Fs = K_s * total_head * soilIMD[i] / (ia - K_s);
  if (soilFu[i] > Fs) // if (_h > Fs)
  {
    Sat[i] = 1;
    dF = grnampt_getSatInfil(soilFu, soilIMD, Sat, _h, K_s,
                             Fumax, total_head, i, tstep);
    return dF;
  }

  if (soilFu[i] + ia * tstep < Fs)
  {
    dF = ia * tstep;
    soilFu[i] += dF;

    return dF;
  }

  scalar_t ts = tstep - (Fs - soilFu[i]) / ia;
  if (ts <= 0.0)
    ts = 0.0;

  scalar_t c1 = total_head * soilIMD[i];
  scalar_t F2 = grnampt_getF2(soilFu[i], c1, K_s, ts);

  if (F2 > Fs + ia * ts)
    F2 = Fs + ia * ts;

  dF = F2 - soilFu[i];
  soilFu[i] = F2;
  Sat[i] = 1;
  return dF;
}

template <typename scalar_t>
__global__ void lidInfiltrationCalculation_kernel(
    int N, int32_t *__restrict__ wetMask_sorted, scalar_t *__restrict__ h_update,
    uint8_t *__restrict__ landuseMask, uint8_t *__restrict__ lidMask, 
    int n_landuse, int num_lid_types, int32_t *__restrict__ Sat,
    scalar_t *__restrict__ h, scalar_t *__restrict__ df, 
    scalar_t *__restrict__ soilInfilKs, scalar_t *__restrict__ soilInfilS,
    scalar_t *__restrict__ soilInfilIMDmax, scalar_t *__restrict__ soilInfilLu,
    scalar_t *__restrict__ soilFu, scalar_t *__restrict__ soilIMD, 
    scalar_t *__restrict__ SoilPara, scalar_t *__restrict__ SurPara,
    scalar_t *__restrict__ cumuSurfaceWaterDepth, scalar_t *__restrict__ areaMask,
    scalar_t *__restrict__ dt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        int32_t i = wetMask_sorted[j];
        uint8_t lidtype = lidMask[i];
        scalar_t areaRatio = areaMask[i];

        if (lidtype < 1 || lidtype == 7) return;

        uint8_t landindex = landuseMask[i];
        int status = Sat[i];
        scalar_t K_s = soilInfilKs[landindex];
        scalar_t Fumax = soilInfilIMDmax[landindex] * soilInfilLu[landindex];
        scalar_t suction_head = soilInfilS[landindex];
        scalar_t tstep = dt[0];
        int lidparaIndex = landindex - n_landuse;

        // Step 1: calculate available water for infiltration
        scalar_t _h = 0.0; // available water for infiltration

        if (lidtype == 3) // green roof
        {
            scalar_t bermHeight = SurPara[lidparaIndex + 0 * num_lid_types];
            // Include both stored water and new inflow
            scalar_t bermWater = cumuSurfaceWaterDepth[i]; // Already stored (local depth)
            scalar_t newInflow = h_update[i] + h[i]; // New water (grid depth = local depth)
            scalar_t totalWater = bermWater + newInflow;

            // available for infiltration (limited by berm capacity)
            if (bermHeight > 0.0)
            {
                _h = min(totalWater, bermHeight);
            }
            else
            {
                _h = totalWater;
            }
            _h = max(_h, 0.0);
        }
        else
        {
            // Other LID types (e.g., permeable pavement, bioretention)
            _h = h_update[i] + h[i]; // New water (grid depth = local depth)
            _h = max(_h, 0.0);
        }

        if (_h < ZERO) {
            df[i] = 0.0;
            return;
        }

        // Step 2: calculate infiltration (Green-Ampt)
        scalar_t total_head = suction_head + _h;
        if (K_s > 0.0)
        {
            if (status == 1)
            {
                df[i] = grnampt_getSatInfil(soilFu, soilIMD, Sat, _h, K_s,
                                           Fumax, total_head, i, tstep);
            }
            else
            {
                df[i] = grnampt_getUnsatInfil(soilFu, soilIMD, Sat, _h, K_s,
                                             Fumax, total_head, i, tstep);
            }
        }
        else
        {
            df[i] = 0.0;
        }
        df[i] = min(df[i], _h);  // Can't infiltrate more than available

        // Step 3: update water distribution
        if (lidtype == 3) // green roof
        {
            scalar_t bermHeight = SurPara[lidparaIndex + 0 * num_lid_types]; 
           // Water balance (all in local depth)
            scalar_t bermWater = cumuSurfaceWaterDepth[i];
            scalar_t newInflow = h[i] + h_update[i];
            scalar_t totalWater = bermWater + newInflow;
            scalar_t infiltrated = df[i];
            scalar_t remaining = totalWater - infiltrated;
            // Handle overflow
            if (bermHeight > 0.0 && remaining > bermHeight)
            {
                scalar_t overflow = remaining - bermHeight;
                cumuSurfaceWaterDepth[i] = bermHeight;
                h_update[i] = overflow - h[i]; // overflow to grid
                
            }
            else // No overflow: all water stays in berm
            {
                cumuSurfaceWaterDepth[i] = max(remaining, 0.0);
                h_update[i] = -h[i];
            }
        }
        else // EMBEDDED LIDs
        {
           // Simply remove infiltrated water from grid
            h_update[i] -= df[i];
            cumuSurfaceWaterDepth[i] = 0.0;
        }
    }
}

template <typename scalar_t>
__global__ void lidIMDini_kernel(
    int N, uint8_t *__restrict__ landuse,
    scalar_t *__restrict__ soilInfilIMDmax,
    scalar_t *__restrict__ soilIMD, scalar_t *__restrict__ cumuSoilMoisture,
    scalar_t *__restrict__ cumuStorageWaterDepth, scalar_t *__restrict__ Stor,
    scalar_t *__restrict__ soilMoi)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N)
  {
    uint8_t land_type = landuse[j];
    // printf("%f\t%f\t%f\n",soilInfilIMDmax[3],soilInfilIMDmax[4],soilInfilIMDmax[6]);
    // printf("%d\t%d\t%f\n",land_type,j,soilInfilIMDmax[land_type]);
    soilIMD[j] = soilInfilIMDmax[land_type];
    cumuSoilMoisture[j] = soilMoi[land_type];
    cumuStorageWaterDepth[j] = 0.0 * Stor[land_type]; // Asumue initial saturate rate = 0
    // printf("%d\t%f\t",j,soilIMD[j]);
  }
}

void lidIMDinitial_cuda(at::Tensor landuse, at::Tensor soilIMD,
                        at::Tensor soilInfilIMDmax, at::Tensor cumuSoilMoisture,
                        at::Tensor cumuStorageWaterDepth, at::Tensor Stor,
                        at::Tensor soilMoi)
{
  const int N = soilIMD.numel();
  at::cuda::CUDAGuard device_guard(soilIMD.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      soilInfilIMDmax.type(), "lidIMDinitial_cuda", ([&]
                                                     { lidIMDini_kernel<
                                                           scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
                                                           N, landuse.data<uint8_t>(), soilInfilIMDmax.data<scalar_t>(),
                                                           soilIMD.data<scalar_t>(), cumuSoilMoisture.data<scalar_t>(),
                                                           cumuStorageWaterDepth.data<scalar_t>(), Stor.data<scalar_t>(), soilMoi.data<scalar_t>()); }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}

void lidInfiltrationCalculation_cuda(
    at::Tensor wetMask_sorted, at::Tensor h_update, at::Tensor landuseMask, at::Tensor lidMask,
    int n_landuse, at::Tensor Sat, at::Tensor h, at::Tensor df, at::Tensor soilInfilKs, at::Tensor soilInfilS,
    at::Tensor soilInfilIMDmax, at::Tensor soilInfilLu, at::Tensor soilFu,
    at::Tensor soilIMD, at::Tensor SoilPara, at::Tensor SurPara, at::Tensor cumuSurfaceWaterDepth, at::Tensor areaMask, at::Tensor dt)
{
  const int N = wetMask_sorted.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  int thread_0 = 256;
  int block_0 = (N + 256 - 1) / 256;
  int num_lid_types = SoilPara.size(1);  // SoilPara shape: [7, n_lid_types]]

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "lidInfiltrationCalculation_cuda", ([&]
                                                    { lidInfiltrationCalculation_kernel<
                                                          scalar_t><<<block_0, thread_0, 0, stream>>>(
                                                          N, wetMask_sorted.data<int32_t>(), h_update.data<scalar_t>(),
                                                          landuseMask.data<uint8_t>(), lidMask.data<uint8_t>(),
                                                          n_landuse, num_lid_types, Sat.data<int>(), h.data<scalar_t>(), df.data<scalar_t>(),
                                                          soilInfilKs.data<scalar_t>(), soilInfilS.data<scalar_t>(),
                                                          soilInfilIMDmax.data<scalar_t>(), soilInfilLu.data<scalar_t>(),
                                                          soilFu.data<scalar_t>(), soilIMD.data<scalar_t>(), SoilPara.data<scalar_t>(), SurPara.data<scalar_t>(),
                                                          cumuSurfaceWaterDepth.data<scalar_t>(), areaMask.data<scalar_t>(), dt.data<scalar_t>()); }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}