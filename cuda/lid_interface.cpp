#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "lid.h"
namespace py = pybind11;

// void lidParaIni(double dt);
//void lidParaIni_cuda(at::Tensor h, at::Tensor qx, at::Tensor qy, at::Tensor dt);

// CUDA forward declarations
//void readlidPara_cuda(int N, int M);
//void lidCalculation_cuda(int N, at::Tensor h, at::Tensor dt);

TLidProc *LidProc;

void lid_initialize(TLidProc *LidProc);
void readlidPara(TLidProc *LidProc, int N, int M);
void lidcalculation(double *ptrh, double *ptrh_input, double *ptrqx_input,
                       double *ptrqy_input, double dt, double dx, double rainfall, TLidProc *LidProc);
double lid_wrtReport(TLidProc *LidProc, double t);

void lidPara(int N, int M)
{
  LidProc = (TLidProc *)malloc(M * N * sizeof(TLidProc));
  readlidPara(LidProc, N, M);
  lid_initialize(LidProc);
}

py::array_t<double> lidCal(py::array_t<double> &h_input, py::array_t<double> &qx_input,
                           py::array_t<double> &qy_input, double dt, double dx, double rainfall)
{
  py::buffer_info buf_h_input = h_input.request(); // h input buf pointer
  py::buffer_info buf_qx_input = qx_input.request();
  py::buffer_info buf_qy_input = qy_input.request();
  auto h = py::array_t<double>(buf_h_input.size); // new calculated h array
  py::buffer_info buf_h = h.request();            //allocate memory
  double *ptrh_input = (double *)buf_h_input.ptr; // C++ pointer to h input pointer
  double *ptrqx_input = (double *)buf_qx_input.ptr;
  double *ptrqy_input = (double *)buf_qy_input.ptr;
  double *ptrh = (double *)buf_h.ptr; // C++ pointer to new calculated h pointer
  lidcalculation(ptrh, ptrh_input, ptrqx_input, ptrqy_input, dt, dx, rainfall, LidProc);
  return h;
}

void lidOutput(double t)
{
  lid_wrtReport(LidProc,t);
}

// void lidCal(int N, at::Tensor h, at::Tensor dt)
// {
//   CHECK_INPUT(h);
//   CHECK_INPUT(dt);
//   lidCalculation_cuda(N, h, dt);
// }

// void lidCal(int N, at::Tensor h, at::Tensor dt)
// {
//   CHECK_INPUT(h);
//   CHECK_INPUT(dt);
//   lidCalculation_cuda(N, h, dt);
// }

PYBIND11_MODULE(lid, m)
{
  m.def("lidPara", &lidPara, "read lid parameters");
  m.def("lidCal", &lidCal, "calculate the lid hdro");
  m.def("lidOutput", &lidOutput, "output lid results");
}