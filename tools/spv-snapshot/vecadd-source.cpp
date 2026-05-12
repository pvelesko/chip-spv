#include <hip/hip_runtime.h>
__global__ void VecAddKernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() { return 0; }
