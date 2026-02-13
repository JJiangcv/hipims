#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
// void lidInfiltrationCalculation_cuda(
//     at::Tensor wetMask, at::Tensor h_update, at::Tensor landuseMask, at::Tensor lidMask,
//     at::Tensor landuse_index,  at::Tensor lidmask_index, at::Tensor Sat, at::Tensor h, at::Tensor df, at::Tensor soilInfilKs, at::Tensor soilInfilS,
//     at::Tensor soilInfilIMDmax, at::Tensor soilInfilLu, at::Tensor soilFu,
//     at::Tensor soilIMD, at::Tensor SoilPara, at::Tensor SurPara, at::Tensor cumuSurfaceWaterDepth, at::Tensor dt);

void lidInfiltrationCalculation_cuda(
    at::Tensor wetMask_sorted, at::Tensor h_update, at::Tensor landuseMask, at::Tensor lidMask,
    int n_landuse, at::Tensor Sat, at::Tensor h, at::Tensor df, at::Tensor soilInfilKs, at::Tensor soilInfilS,
    at::Tensor soilInfilIMDmax, at::Tensor soilInfilLu, at::Tensor soilFu,
    at::Tensor soilIMD, at::Tensor SoilPara, at::Tensor SurPara, at::Tensor cumuSurfaceWaterDepth, at::Tensor areaMask, at::Tensor dt);

void lidIMDinitial_cuda(at::Tensor landuse, at::Tensor soilIMD,
                        at::Tensor soilInfilIMDmax, at::Tensor cumuSoilMoisture,
                        at::Tensor cumuStorageWaterDepth,
                        at::Tensor Stor, at::Tensor soilMoi);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void lid_InfiltrationCalculation(
  at::Tensor wetMask_sorted, at::Tensor h_update, at::Tensor landuseMask, at::Tensor lidMask,
  int n_landuse, at::Tensor Sat,
  at::Tensor h, at::Tensor df, at::Tensor soilInfilKs, at::Tensor soilInfilS,
  at::Tensor soilInfilIMDmax, at::Tensor soilInfilLu, at::Tensor soilFu,
  at::Tensor soilIMD, at::Tensor SoilPara, at::Tensor SurPara, at::Tensor cumuSurfaceWaterDepth, at::Tensor areaMask, at::Tensor dt)
{
  CHECK_INPUT(wetMask_sorted);
  CHECK_INPUT(h_update);
  CHECK_INPUT(landuseMask);
  CHECK_INPUT(lidMask);
  CHECK_INPUT(Sat);
  CHECK_INPUT(h);
  CHECK_INPUT(df);
  CHECK_INPUT(soilInfilKs);
  CHECK_INPUT(soilInfilS);
  CHECK_INPUT(soilInfilIMDmax);
  CHECK_INPUT(soilInfilLu);
  CHECK_INPUT(soilIMD);
  CHECK_INPUT(soilFu);
  CHECK_INPUT(SoilPara);
  CHECK_INPUT(SurPara);
  CHECK_INPUT(cumuSurfaceWaterDepth);
  CHECK_INPUT(dt);
  

  lidInfiltrationCalculation_cuda(
      wetMask_sorted, h_update, landuseMask, lidMask, n_landuse, Sat, h, df, soilInfilKs, soilInfilS,
      soilInfilIMDmax, soilInfilLu, soilFu, soilIMD, SoilPara, SurPara, cumuSurfaceWaterDepth,areaMask, dt);
}

// void lid_InfiltrationCalculation(
//     at::Tensor wetMask, at::Tensor h_update, at::Tensor landuseMask, at::Tensor lidMask,
//     at::Tensor landuse_index,  at::Tensor lidmask_index, at::Tensor Sat,
//     at::Tensor h, at::Tensor df, at::Tensor soilInfilKs, at::Tensor soilInfilS,
//     at::Tensor soilInfilIMDmax, at::Tensor soilInfilLu, at::Tensor soilFu,
//     at::Tensor soilIMD, at::Tensor SoilPara, at::Tensor SurPara, at::Tensor cumuSurfaceWaterDepth, at::Tensor dt)
// {
//   CHECK_INPUT(wetMask);
//   CHECK_INPUT(lidMask);
//   CHECK_INPUT(landuseMask);
//   CHECK_INPUT(h);
//   CHECK_INPUT(h_update);
//   CHECK_INPUT(dt);
//   CHECK_INPUT(soilInfilKs);
//   CHECK_INPUT(soilInfilS);
//   CHECK_INPUT(soilInfilIMDmax);
//   CHECK_INPUT(soilInfilLu);
//   CHECK_INPUT(soilIMD);
//   CHECK_INPUT(soilFu);
//   CHECK_INPUT(Sat);

//   lidInfiltrationCalculation_cuda(
//       wetMask, h_update, landuseMask, lidMask, landuse_index, lidmask_index, Sat, h, df, soilInfilKs, soilInfilS,
//       soilInfilIMDmax, soilInfilLu, soilFu, soilIMD, SoilPara, SurPara, cumuSurfaceWaterDepth, dt);
// }

void lidIMDinitial(at::Tensor landuse, at::Tensor soilInfilIMDmax,
                   at::Tensor soilIMD, at::Tensor cumuSoilMoisture,
                   at::Tensor cumuStorageWaterDepth,
                   at::Tensor Stor, at::Tensor soilMoi)
{
  CHECK_INPUT(landuse);
  CHECK_INPUT(soilInfilIMDmax);
  CHECK_INPUT(soilIMD);
  CHECK_INPUT(cumuSoilMoisture);
  CHECK_INPUT(cumuStorageWaterDepth);
  CHECK_INPUT(Stor);
  CHECK_INPUT(soilMoi);

  lidIMDinitial_cuda(landuse, soilIMD, soilInfilIMDmax, cumuSoilMoisture,
                     cumuStorageWaterDepth, Stor, soilMoi);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("lidIMD_ini", &lidIMDinitial, "Lid_IMD_initialize, CUDA version");
  m.def("addLidInfiltrationSource", &lid_InfiltrationCalculation,
        "Lid_Infiltration_Calculation, CUDA version");
}