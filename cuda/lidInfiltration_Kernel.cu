// #include "gpu.cuh"
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdint.h>
// #include <torch/extension.h>
// #include <math.h>
// #include <stdio.h>

// #define ZERO 1.E-10 // Effective zero value

// template <typename scalar_t>
// __device__ __forceinline__ double grnampt_getF2(scalar_t f1, scalar_t c1,
//                                                 scalar_t K_s, scalar_t ts)
// {
//   // printf("grnampt_getF2\n");
//   scalar_t f2 = f1;
//   scalar_t f2min = 0.0;
//   scalar_t df2 = 0.0;
//   scalar_t c2;
//   // --- find min. infil. volume
//   f2min = f1 + K_s * ts;

//   // --- use min. infil. volume for 0 moisture deficit
//   if (c1 == 0.0)
//   {
//     // printf("3\n");
//     return f2min;
//   }

//   // --- use direct form of G-A equation for small time steps
//   //     and c1/f1 < 100
//   if (ts < 10.0 && f1 > 0.01 * c1)
//   {
//     // printf("1\n");
//     f2 = f1 + K_s * (1.0 + c1 / f1) * ts;
//     return max(f2, f2min);
//   }
//   // --- use Newton-Raphson method to solve integrated G-A equation
//   //     (convergence limit reduced from that used in previous releases)
//   c2 = c1 * log((f1 + c1)) - K_s * ts;
//   for (int i = 1; i <= 20; i++)
//   {
//     // printf("2\n");
//     df2 = (f2 - f1 - c1 * log(f2 + c1) + c2) / (1.0 - c1 / (f2 + c1));
//     if (abs(df2) < 0.00001)
//     {
//       return max(f2, f2min);
//     }
//     f2 -= df2;
//   }

//   return f2min;
// }

// template <typename scalar_t>
// __device__ __forceinline__ double
// grnampt_getSatInfil(scalar_t *soilFu,
//                     scalar_t *soilIMD, int *Sat, scalar_t _h, scalar_t K_s,
//                     scalar_t Fumax, scalar_t total_head, int i,
//                     scalar_t tstep)
// {
//   scalar_t ia = _h / tstep;
//   scalar_t c1 = total_head * soilIMD[i];
//   scalar_t F2 = grnampt_getF2(soilFu[i], c1, K_s, tstep); // rnampt_getF2(_h, c1, K_s, tstep)
//   scalar_t dF = F2 - soilFu[i];
//   // --- all available water infiltrates -- set saturated state to false

//   if (dF > ia * tstep)
//   {
//     dF = ia * tstep;
//     Sat[i] = 0;
//   }
//   // --- update total infiltration and upper zone moisture deficit
//   // cumulativedepth += dF;
//   soilFu[i] += dF;
//   soilFu[i] = min(soilFu[i], Fumax);
//   return dF;
// }

// template <typename scalar_t>
// __device__ __forceinline__ double
// grnampt_getUnsatInfil(scalar_t *soilFu,
//                       scalar_t *soilIMD, int *Sat, scalar_t _h, scalar_t K_s,
//                       scalar_t Fumax, scalar_t total_head, int i,
//                       scalar_t tstep)
// {
//   scalar_t ia = _h / tstep;
//   scalar_t dF = 0.0;
//   // printf("_h:%f\tt:%f\tia:%f\t", _h, tstep, ia);
//   if (ia < ZERO)
//   {
//     ia = 0;
//   }
//   if (ia <= K_s)
//   {
//     // printf("ia<=Ks\n");
//     dF = ia * tstep;
//     soilFu[i] += dF;
//     // soilFu[i] = min(soilFu[i], Fumax);
//     //  printf("soilFu[%d]%f\n", i,soilFu[i]);
//     return dF;
//   }
//   scalar_t Fs = K_s * total_head * soilIMD[i] / (ia - K_s);
//   // printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t",i, K_s, total_head, soilIMD[i], ia,
//   // printf("soilFu[i]%f\n", soilFu[i]);
//   // K_s,
//   if (soilFu[i] > Fs) // if (_h > Fs)
//   {
//     Sat[i] = 1;
//     dF = grnampt_getSatInfil(soilFu, soilIMD, Sat, _h, K_s,
//                              Fumax, total_head, i, tstep);
//     // printf("DF%f\n", dF);
//     return dF;
//   }

//   if (soilFu[i] + ia * tstep < Fs)
//   {
//     dF = ia * tstep;
//     // cumulativedepth += dF;
//     soilFu[i] += dF;
//     // soilFu[i] = min(soilFu[i], Fumax);
//     // printf("DF%f\n", dF);
//     return dF;
//   }

//   // scalar_t ts = tstep - (Fs - _h) / ia;
//   scalar_t ts = tstep - (Fs - soilFu[i]) / ia;
//   if (ts <= 0.0)
//     ts = 0.0;

//   scalar_t c1 = total_head * soilIMD[i];
//   scalar_t F2 = grnampt_getF2(soilFu[i], c1, K_s, ts);
//   // printf("%f\t",F2);
//   if (F2 > Fs + ia * ts)
//     F2 = Fs + ia * ts;

//   dF = F2 - soilFu[i];
//   // cumulativedepth = F2;
//   // soilFu[i] += dF;
//   soilFu[i] = F2;
//   // soilFu[i] = min(soilFu[i], Fumax);
//   Sat[i] = 1;
//   // printf("dF:%f\t soilFu:%f\t Fumax%f\t",dF,soilFu[i],Fumax);
//   return dF;
// }

// template <typename scalar_t>
// __global__ void lidInfiltrationCalculation_kernel(
//     int N, int32_t *__restrict__ wetMask, scalar_t *__restrict__ h_update,
//     uint8_t *__restrict__ landuseMask, uint8_t *__restrict__ lidMask, uint8_t num, uint8_t Index, int32_t *__restrict__ Sat,
//     scalar_t *__restrict__ h, scalar_t *__restrict__ df, scalar_t *__restrict__ soilInfilKs,
//     scalar_t *__restrict__ soilInfilS, scalar_t *__restrict__ soilInfilIMDmax,
//     scalar_t *__restrict__ soilInfilLu, scalar_t *__restrict__ soilFu,
//     scalar_t *__restrict__ soilIMD, scalar_t *__restrict__ SoilPara, scalar_t *__restrict__ SurPara, scalar_t *__restrict__ cumuSurfaceWaterDepth,
//     scalar_t *__restrict__ dt)
// {
//   // get the index of cell
//   int j = blockIdx.x * blockDim.x + threadIdx.x;
//   if (j < N)
//   {
//     int32_t i = wetMask[j];
//     uint8_t landindex = landuseMask[i];
//     if (landindex == 129)
//     {
//       printf("here\n");
//       return;
//       // printf("landindex:%d\t land_type:%d\n", landindex, land_type);
//     }
//     uint8_t land_type = lidMask[i];
//     int sat_status = Sat[i];
//     scalar_t K_s = soilInfilKs[landindex];
//     scalar_t Fumax = soilInfilIMDmax[landindex] * soilInfilLu[landindex];
//     int8_t lidparaIndex = landindex - Index;
//     scalar_t surfaceThickness = SurPara[lidparaIndex + 0 * num];
//     // scalar_t surfacevoid = SurPara[landindex];
//     //  if (landindex == 129)
//     //  {
//     //    printf("here\n");
//     //    return;
//     //    // printf("landindex:%d\t land_type:%d\n", landindex, land_type);
//     //  }
//     //  printf("beforeinfil h[%d]:%f\n", i, h[i]); ///Print debug || land_type == 5
//     scalar_t _h;
//     if (land_type == 3)
//     {
//       scalar_t totalSurfaceWater = cumuSurfaceWaterDepth[i] + h[i] + h_update[i];
//       if (surfaceThickness > 0.0)
//       {
//         // 只有berm容量内的水参与渗透
//         _h = min(totalSurfaceWater, surfaceThickness);
//         _h = max(_h, 0.0);
//       }
//       else
//       {
//         // 没有berm时，所有水都可渗透
//         _h = max(totalSurfaceWater, 0.0);
//       }
//     }
//     else if (land_type == 5)  // Permeable pavement - FIXED
//     {
//       // 修复：所有表面水都应该参与渗透
//       // 不应该在h[i]>0时忽略h_update[i]
//       _h = h[i] + h_update[i];
//       _h = max(_h, 0.0);
//     }
//     else
//     {
//       _h = h[i] + h_update[i];
//     }

//     scalar_t total_head = soilInfilS[landindex] + _h;
//     scalar_t tstep = dt[0];
//     scalar_t delta_F = 0.0;
//     // scalar_t cumulativedepth = h[i];
//     if (land_type < 1 || land_type == 129)
//     {
//       return;
//     }
//     // printf("Fumax:%f\n", Fumax);
//     if (land_type == 6)
//     {
//       // delta_F = 0.0;
//       df[i] = 0.0;
//     }
//     else if (K_s > 0.0)
//     {
//       // printf("K_s > 0.0");
//       //  printf("here\n");
//       //  printf("landindex:%d\t land_type:%d\n", landindex, land_type);
//       if (sat_status == 1)
//       {
//         // printf("satinfil");
//         df[i] = grnampt_getSatInfil(soilFu, soilIMD, Sat,
//                                     _h, K_s, Fumax, total_head, i, tstep);
//         // delta_F = 100;
//       }
//       else
//       {
//         // printf("unsatinfil");
//         //  printf("%d\n",sat_status);
//         df[i] = grnampt_getUnsatInfil(soilFu, soilIMD, Sat,
//                                       _h, K_s, Fumax, total_head, i, tstep);
//         // printf("%f\n",df[i]);
//       }
//     }
//     else
//     {
//       delta_F = _h;
//       df[i] = delta_F;
//       // printf(" delta_F:%f\n", delta_F);
//     }
//     // printf("h_update[i]:%f\n", h_update[i]);
//     // h_update[i] -= delta_F;
//     // df[i] = delta_F;
//     // printf("df[%d]:%f\n", i, df[i]);
//     // printf("afterinfil h[%d]:%f\n", i, h[i]); ///print debug
//   }
// }

// template <typename scalar_t>
// __global__ void lidIMDini_kernel(
//     int N, uint8_t *__restrict__ landuse,
//     scalar_t *__restrict__ soilInfilIMDmax,
//     scalar_t *__restrict__ soilIMD, scalar_t *__restrict__ cumuSoilMoisture,
//     scalar_t *__restrict__ cumuStorageWaterDepth, scalar_t *__restrict__ Stor,
//     scalar_t *__restrict__ soilMoi)
// {
//   int j = blockIdx.x * blockDim.x + threadIdx.x;
//   if (j < N)
//   {
//     uint8_t land_type = landuse[j];
//     // printf("%f\t%f\t%f\n",soilInfilIMDmax[3],soilInfilIMDmax[4],soilInfilIMDmax[6]);
//     // printf("%d\t%d\t%f\n",land_type,j,soilInfilIMDmax[land_type]);
//     soilIMD[j] = soilInfilIMDmax[land_type];
//     cumuSoilMoisture[j] = soilMoi[land_type];
//     cumuStorageWaterDepth[j] = 0.0 * Stor[land_type]; // Asumue initial saturate rate = 0
//     // printf("%d\t%f\t",j,soilIMD[j]);
//   }
// }

// void lidIMDinitial_cuda(at::Tensor landuse, at::Tensor soilIMD,
//                         at::Tensor soilInfilIMDmax, at::Tensor cumuSoilMoisture,
//                         at::Tensor cumuStorageWaterDepth, at::Tensor Stor,
//                         at::Tensor soilMoi)
// {
//   const int N = soilIMD.numel();
//   at::cuda::CUDAGuard device_guard(soilIMD.device());
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   AT_DISPATCH_FLOATING_TYPES(
//       soilInfilIMDmax.type(), "lidIMDinitial_cuda", ([&]
//                                                      { lidIMDini_kernel<
//                                                            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
//                                                            N, landuse.data<uint8_t>(), soilInfilIMDmax.data<scalar_t>(),
//                                                            soilIMD.data<scalar_t>(), cumuSoilMoisture.data<scalar_t>(),
//                                                            cumuStorageWaterDepth.data<scalar_t>(), Stor.data<scalar_t>(), soilMoi.data<scalar_t>()); }));
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess)
//     printf("Error in load_textures: %s\n", cudaGetErrorString(err));
// }

// void lidInfiltrationCalculation_cuda(
//     at::Tensor wetMask, at::Tensor h_update, at::Tensor landuseMask, at::Tensor lidMask,
//     at::Tensor landuse_index, at::Tensor lidmask_index, at::Tensor Sat, at::Tensor h, at::Tensor df, at::Tensor soilInfilKs, at::Tensor soilInfilS,
//     at::Tensor soilInfilIMDmax, at::Tensor soilInfilLu, at::Tensor soilFu,
//     at::Tensor soilIMD, at::Tensor SoilPara, at::Tensor SurPara, at::Tensor cumuSurfaceWaterDepth, at::Tensor dt)
// {
//   const int N = wetMask.numel();
//   // printf("N:%d\n",N);
//   at::cuda::CUDAGuard device_guard(h.device());
//   int thread_0 = 256;
//   int block_0 = (N + 256 - 1) / 256;
//   uint8_t num = lidmask_index.numel();
//   uint8_t Index = landuse_index.numel();
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   AT_DISPATCH_FLOATING_TYPES(
//       h.type(), "lidInfiltrationCalculation_cuda", ([&]
//                                                     { lidInfiltrationCalculation_kernel<
//                                                           scalar_t><<<block_0, thread_0, 0, stream>>>(
//                                                           N, wetMask.data<int32_t>(), h_update.data<scalar_t>(),
//                                                           landuseMask.data<uint8_t>(), lidMask.data<uint8_t>(),
//                                                           num, Index, Sat.data<int>(), h.data<scalar_t>(), df.data<scalar_t>(),
//                                                           soilInfilKs.data<scalar_t>(), soilInfilS.data<scalar_t>(),
//                                                           soilInfilIMDmax.data<scalar_t>(), soilInfilLu.data<scalar_t>(),
//                                                           soilFu.data<scalar_t>(), soilIMD.data<scalar_t>(), SoilPara.data<scalar_t>(), SurPara.data<scalar_t>(),
//                                                           cumuSurfaceWaterDepth.data<scalar_t>(), dt.data<scalar_t>()); }));
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess)
//     printf("Error in load_textures: %s\n", cudaGetErrorString(err));
// }


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
  // cumulativedepth += dF;
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
  // printf("_h:%f\tt:%f\tia:%f\t", _h, tstep, ia);
  if (ia < ZERO)
  {
    ia = 0;
  }
  if (ia <= K_s)
  {
    // printf("ia<=Ks\n");
    dF = ia * tstep;
    soilFu[i] += dF;
    // soilFu[i] = min(soilFu[i], Fumax);
    //  printf("soilFu[%d]%f\n", i,soilFu[i]);
    return dF;
  }
  scalar_t Fs = K_s * total_head * soilIMD[i] / (ia - K_s);
  // printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t",i, K_s, total_head, soilIMD[i], ia,
  // printf("soilFu[i]%f\n", soilFu[i]);
  // K_s,
  if (soilFu[i] > Fs) // if (_h > Fs)
  {
    Sat[i] = 1;
    dF = grnampt_getSatInfil(soilFu, soilIMD, Sat, _h, K_s,
                             Fumax, total_head, i, tstep);
    // printf("DF%f\n", dF);
    return dF;
  }

  if (soilFu[i] + ia * tstep < Fs)
  {
    dF = ia * tstep;
    // cumulativedepth += dF;
    soilFu[i] += dF;
    // soilFu[i] = min(soilFu[i], Fumax);
    // printf("DF%f\n", dF);
    return dF;
  }

  // scalar_t ts = tstep - (Fs - _h) / ia;
  scalar_t ts = tstep - (Fs - soilFu[i]) / ia;
  if (ts <= 0.0)
    ts = 0.0;

  scalar_t c1 = total_head * soilIMD[i];
  scalar_t F2 = grnampt_getF2(soilFu[i], c1, K_s, ts);
  // printf("%f\t",F2);
  if (F2 > Fs + ia * ts)
    F2 = Fs + ia * ts;

  dF = F2 - soilFu[i];
  // cumulativedepth = F2;
  // soilFu[i] += dF;
  soilFu[i] = F2;
  // soilFu[i] = min(soilFu[i], Fumax);
  Sat[i] = 1;
  // printf("dF:%f\t soilFu:%f\t Fumax%f\t",dF,soilFu[i],Fumax);
  return dF;
}

template <typename scalar_t>
__global__ void lidInfiltrationCalculation_kernel(
    int N, int32_t *__restrict__ wetMask_sorted, scalar_t *__restrict__ h_update,
    uint8_t *__restrict__ landuseMask, uint8_t *__restrict__ lidMask, int n_landuse, int num_lid_types, 
    int32_t *__restrict__ Sat,
    scalar_t *__restrict__ h, scalar_t *__restrict__ df, scalar_t *__restrict__ soilInfilKs,
    scalar_t *__restrict__ soilInfilS, scalar_t *__restrict__ soilInfilIMDmax,
    scalar_t *__restrict__ soilInfilLu, scalar_t *__restrict__ soilFu,
    scalar_t *__restrict__ soilIMD, scalar_t *__restrict__ SoilPara, scalar_t *__restrict__ SurPara, 
    scalar_t *__restrict__ cumuSurfaceWaterDepth, scalar_t *__restrict__ areaMask,
    scalar_t *__restrict__ dt)
{
  // get the index of cell
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N)
  {
    int32_t i = wetMask_sorted[j];
    uint8_t lidtype  = lidMask[i];
    scalar_t areaRatio = areaMask[i];
    // Early exit for non-LID cells
    if (lidtype < 1 || lidtype == 129 || lidtype == 6 || lidtype == 7) {
        return;
    }

    uint8_t landindex = landuseMask[i];
    int sat_status = Sat[i];

    // Get soil parameters
    scalar_t K_s = soilInfilKs[landindex];
    scalar_t Fumax = soilInfilIMDmax[landindex] * soilInfilLu[landindex];
    scalar_t suction_head = soilInfilS[landindex];
    scalar_t tstep = dt[0];
    // Calculate parameter index for LID-specific arrays
    int lidparaIndex = landindex - n_landuse;
    // ================================================================
    // STEP 1: Calculate available water for infiltration (_h)
    // ================================================================
    scalar_t _h = 0.0;
    scalar_t overflow = 0.0;

    if(lidtype==3) // green roof
    {
      // Get berm height (surface thickness)
      scalar_t bermHeight = SurPara[lidparaIndex + 0 * num_lid_types];
      // Calculate total water
      scalar_t freeWater = h[i] + h_update[i];
      scalar_t pondedWater = cumuSurfaceWaterDepth[i];
      scalar_t totalWater = freeWater + pondedWater;
      if (bermHeight > 0.0 && totalWater > bermHeight) {
        // overflow occurs
        _h = bermHeight;                            // water up to berm height
        overflow = totalWater - bermHeight;         // overflow water
      } 
      else {
        // no overflow
        _h = totalWater;
        overflow = 0.0;
      }
      _h = max(_h, 0.0);
    }
    else // no over flow
    {
      _h = h[i] + h_update[i];
      _h = max(_h, 0.0);
      overflow = 0.0;
    }
    
    scalar_t total_head = suction_head + _h;
    if (_h < ZERO) {
            df[i] = 0.0;
            return;
        }

    // ================================================================
    // STEP 2: calculate infiltration based on soil saturation status
    // ================================================================
    if (lidtype == 6)
    {
      df[i] = 0.0;
    }
    else if (K_s > 0.0)
    {
      if (sat_status == 1)
      {
        df[i] = grnampt_getSatInfil(soilFu, soilIMD, Sat,
                                    _h, K_s, Fumax, total_head, i, tstep);
      }
      else
      {
        df[i] = grnampt_getUnsatInfil(soilFu, soilIMD, Sat,
                                      _h, K_s, Fumax, total_head, i, tstep);
      }
    }
    else
    {
      df[i] = _h;
    }

    df[i] = min(df[i], _h);
    
    df[i] = df[i] * areaRatio;

    // ================================================================
    // STEP 3: update water depths after infiltration
    // ================================================================

    if (lidtype == 3) // green roof
    {
      scalar_t bermHeight = SurPara[lidparaIndex + 0 * num_lid_types];
      // available water after infiltration minus overflow
      scalar_t remainingInBerm = _h - df[i] / areaRatio;

      // Update free water and ponded water
      if(bermHeight > 0.0) {
          // 更新berm内的水
          cumuSurfaceWaterDepth[i] = max(remainingInBerm, 0.0) * areaRatio;
          h_update[i] = overflow *areaRatio - h[i]; // free water is overflow minus original free water
      } else {
          // no berm, all remaining water is free water
          h_update[i] = remainingInBerm*areaRatio - h[i];
          cumuSurfaceWaterDepth[i] = 0.0;
      }
    }
    else
    {
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