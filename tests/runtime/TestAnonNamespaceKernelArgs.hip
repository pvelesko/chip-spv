// Scalar and pointer arguments of a kernel in an anonymous namespace reach
// the right parameters.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {
__global__ void copyStrided(int64_t N, const float *X, int64_t IncX, float *Y,
                            int64_t IncY) {
  for (int64_t I = threadIdx.x; I < N; I += blockDim.x)
    Y[I * IncY] = X[I * IncX];
}
} // namespace

int main() {
  constexpr int64_t N = 64;
  float *X, *Y, H[2 * N];
  for (int64_t I = 0; I < 2 * N; I++)
    H[I] = I + 1;
  (void)hipMalloc(&X, sizeof(H));
  (void)hipMalloc(&Y, N * sizeof(float));
  (void)hipMemcpy(X, H, sizeof(H), hipMemcpyHostToDevice);
  copyStrided<<<1, 32>>>(N, X, 2, Y, 1);
  (void)hipMemcpy(H, Y, N * sizeof(float), hipMemcpyDeviceToHost);
  for (int64_t I = 0; I < N; I++)
    if (H[I] != 2 * I + 1) {
      printf("FAILED: Y[%ld] = %g, expected %ld\n", (long)I, H[I],
             (long)(2 * I + 1));
      return 1;
    }
  printf("PASSED\n");
  return 0;
}
