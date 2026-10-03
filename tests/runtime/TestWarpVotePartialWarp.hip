// __all and __any vote over the lanes a warp really has: a 16-thread block,
// and the partial last warp of a 48-thread block.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void votes(int *Out) {
  int T = threadIdx.x;
  Out[T] = __all(1) | (__any(T == 5) << 1) | (__all(T != 3) << 2);
}

int main() {
  int Bad = 0;
  for (int N : {16, 48}) {
    int *D, H[48];
    (void)hipMalloc(&D, sizeof(H));
    votes<<<1, N>>>(D);
    (void)hipMemcpy(H, D, N * sizeof(int), hipMemcpyDeviceToHost);
    for (int T = 0; T < N; ++T) {
      bool First = T < warpSize;
      int Exp = 1 | (First << 1) | (!First << 2);
      if (H[T] != Exp && Bad++ < 4)
        printf("block %d lane %d: %d, expected %d\n", N, T, H[T], Exp);
    }
    (void)hipFree(D);
  }
  printf("%s\n", Bad ? "FAILED" : "PASSED");
  return Bad != 0;
}
