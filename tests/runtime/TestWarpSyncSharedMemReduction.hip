// Warp lock-step shared-memory reduction: no warp primitive is called, so the
// only thing that makes this kernel correct is that a warp really is warpSize
// work items wide and really does run in lock step.
//
// This is the shape Kokkos' HIP backend uses for a block reduction
// (HIPReductionsFunctor<F,false>::scalar_intra_block_reduction): a barrier-free
// binary tree over warpSize shared-memory slots, then a barrier-free broadcast
// of the warp leader's slot to the whole warp, then the same again across warps.
//
// If the subgroup the kernel is actually compiled for is narrower than
// warpSize, the halves of a warp are separate hardware threads with no ordering
// between them. The upper half's broadcast then overwrites the slot the lower
// half's leader is still about to read, and the leader adds its own partial to
// itself, so the single contribution comes out multiplied.
//
// Two things here exist to make that race land reliably rather than
// occasionally:
//
//  * the helpers are noinline -- out-of-line calls are what tip IGC into
//    choosing a narrower subgroup, and they are what an unoptimized build
//    (CHIP_JIT_FLAGS=-cl-opt-disable) produces for Kokkos;
//  * each stage snapshots its shared slot to global memory. The long-latency
//    store holds the lower half of the warp back far enough that the upper half
//    reaches its broadcast first. Without it the two halves usually happen to
//    interleave harmlessly and the bug hides.
//
// The snapshots are also checked, so a failure says which stage went wrong.
//
// Regression test for CHIP-SPV/chipStar#1409.

#include <hip/hip_runtime.h>
#include <cstdio>

#define NTHREADS 128
#define SNAP_INTRA 0     // shared[y] after the intra-warp tree + broadcast
#define SNAP_FOLD NTHREADS // shared[y] after the cross-warp fold
#define SNAP_SIZE (2 * NTHREADS)

struct Reducer {
  double *snap;
  __device__ void join(double *a, double const *b) const { *a += *b; }
  __device__ void init(double *a) const { *a = 0.0; }
};

__device__ __attribute__((noinline)) void
intraWarpReduction(Reducer const &functor, double *value, bool skipVector,
                   int width) {
  int laneId = (threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  // blockDim.x is 1 here, so this is the same stride as 1 -- but it is a
  // runtime value, which is what keeps the tree a real loop, exactly as in
  // Kokkos. A constant-folded, fully unrolled tree hides the race.
  for (int delta = skipVector ? blockDim.x : 1; delta < width; delta *= 2) {
    if (laneId + delta < width && (laneId % (delta * 2) == 0))
      functor.join(value, value + delta);
  }
  // Broadcast the warp leader's slot to every lane of the warp.
  *value = *(value - laneId);
}

__device__ __attribute__((noinline)) void
intraBlockReduction(Reducer const &functor, double value, double *result,
                    double *shared) {
  unsigned warpId = (threadIdx.y * blockDim.x) / warpSize;
  unsigned numThreads = blockDim.x * blockDim.y;
  double *mySlot = shared + threadIdx.y * blockDim.x + threadIdx.x;
  *mySlot = value;

  intraWarpReduction(functor, mySlot, true,
                     numThreads < (unsigned)warpSize ? numThreads : warpSize);
  functor.snap[SNAP_INTRA + threadIdx.y] = shared[threadIdx.y];
  __syncthreads();

  // One warp folds the per-warp totals together.
  if (warpId == 0u) {
    unsigned stride = (threadIdx.y * blockDim.x + threadIdx.x) * warpSize;
    if (stride < numThreads)
      *mySlot = shared[stride];
    intraWarpReduction(functor, mySlot, false, numThreads / warpSize);
    functor.snap[SNAP_FOLD + threadIdx.y] = shared[threadIdx.y];
    if (threadIdx.x + threadIdx.y == 0u)
      *result = *shared;
  }
}

// REQUIRED (1, N, 1): the reduction indexes shared memory by threadIdx.y, just
// like Kokkos' ParallelReduce<RangePolicy, HIP>.
__global__ __launch_bounds__(NTHREADS) void warpSyncReduce(double *out,
                                                           double *snap) {
  extern __shared__ double shared[];
  Reducer functor{snap};
  unsigned y = threadIdx.y;

  // One single contribution; everything else is the identity.
  functor.init(&shared[y]);
  if (y == 0)
    shared[0] = 5.0;
  __syncthreads();

  double value = shared[y];
  __syncthreads();
  intraBlockReduction(functor, value, out, shared);
}

int main() {
  double *dOut = nullptr, *dSnap = nullptr;
  if (hipMalloc(&dOut, sizeof(double)) != hipSuccess ||
      hipMalloc(&dSnap, SNAP_SIZE * sizeof(double)) != hipSuccess) {
    printf("hipMalloc failed\n");
    return 1;
  }

  int failures = 0;
  // The defect is a data race between hardware threads, so sample it a few
  // times rather than trusting a single launch.
  for (int rep = 0; rep < 8; ++rep) {
    if (hipMemset(dOut, 0, sizeof(double)) != hipSuccess ||
        hipMemset(dSnap, 0, SNAP_SIZE * sizeof(double)) != hipSuccess) {
      printf("hipMemset failed\n");
      return 1;
    }
    hipLaunchKernelGGL(warpSyncReduce, dim3(1, 1, 1), dim3(1, NTHREADS, 1),
                       (NTHREADS + 2) * sizeof(double), 0, dOut, dSnap);
    if (hipDeviceSynchronize() != hipSuccess) {
      printf("launch failed\n");
      return 1;
    }
    double h = -1.0, snap[SNAP_SIZE];
    if (hipMemcpy(&h, dOut, sizeof(double), hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(snap, dSnap, sizeof(snap), hipMemcpyDeviceToHost) !=
            hipSuccess) {
      printf("hipMemcpy failed\n");
      return 1;
    }
    if (h != 5.0) {
      printf("rep %d: reduced %g, expected 5 (over-counted %gx); "
             "after intra-warp shared[0]=%g shared[16]=%g shared[24]=%g\n",
             rep, h, h / 5.0, snap[SNAP_INTRA + 0], snap[SNAP_INTRA + 16],
             snap[SNAP_INTRA + 24]);
      ++failures;
    }
  }
  (void)hipFree(dOut);
  (void)hipFree(dSnap);

  if (failures) {
    printf("FAILED: %d of 8 reps over-counted\n", failures);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
