// Minimal C interface to CLIPPER (github.com/mit-acl/clipper), called from rerank_lib.clipper_matches
// through ctypes (the project's pybind11 module crashed on our Python 3.11 host).
// Build CLIPPER first (cmake -DCLIPPER_ENABLE_SCS_SDR=OFF), then:
//   g++ -O2 -march=native -mavx -mfma -std=c++17 -fPIC -shared clipper_c.cpp -I<clipper>/include \
//       -I<eigen3> -L<clipper>/build -lclipper -Wl,-rpath,<clipper>/build -fopenmp -o libclipper_c.so
// and point CLIPPER_C at libclipper_c.so (default ~/clipper_c/libclipper_c.so).
#include <clipper/clipper.h>
#include <clipper/invariants/euclidean_distance.h>
extern "C" int clipper_select(const double* X1, int n1, const double* X2, int n2, int d,
                              const int* A, int m, double eps, double sigma, int* out) {
  Eigen::Map<const Eigen::MatrixXd> D1(X1, d, n1), D2(X2, d, n2);   // rows of X are points
  Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>> Am(A, m, 2);
  clipper::Association As = Am;
  clipper::invariants::EuclideanDistance::Params ip; ip.epsilon = eps; ip.sigma = sigma;
  auto inv = std::make_shared<clipper::invariants::EuclideanDistance>(ip);
  clipper::Params p; p.rounding = clipper::Params::Rounding::DSD_HEU;
  clipper::CLIPPER cl(inv, p);
  cl.setParallelize(false);
  cl.scorePairwiseConsistency(D1, D2, As);
  cl.solve();
  clipper::Association S = cl.getSelectedAssociations();
  for (int k = 0; k < S.rows(); ++k) { out[2 * k] = S(k, 0); out[2 * k + 1] = S(k, 1); }
  return (int)S.rows();
}
