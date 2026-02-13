#include <torch/extension.h>
#include <vector>

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

void lid_InfiltrationCalculation_new(
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

void lidIMDinitial_new(at::Tensor landuse, at::Tensor soilInfilIMDmax,
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
  m.def("lidIMD_ini_new", &lidIMDinitial_new, "Lid_IMD_initialize_new, CUDA version");
  m.def("addLidInfiltrationSource_new", &lid_InfiltrationCalculation_new,
        "Lid_Infiltration_Calculation_new, CUDA version");
}