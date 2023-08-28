
#include "Filter.h"
#include "Graph.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void getEncoding(int *nodesCount,
                 int *edgesCount,
                 int *edges_labels,
                 int *index1,
                 int *index2,
                 int *encoding) {
  int node_id = blockIdx.x * blockDim.x + threadIdx.x; // thread_id
  if (node_id >= nodesCount[0]) return;
  int start = index2[index1[node_id]];
  int end = index2[index1[node_id + 1]];
  Label_t l;
  for (int i = start; i < end; ++i) {
    l = edges_labels[i];
    printf("l: %d\n", l);
    encoding[node_id * edgesCount[0] + l] += 1;
  }
//  cudaThreadSynchronize();
//  cudaDeviceSynchronize();
  __syncthreads();
  // encode to 00, 01, 11
  // stable complexity: O(m) per thread. Avoiding too slow in case of high degree.
  for (int i = node_id * edgesCount[0]; i < (node_id + 1) * edgesCount[0]; ++i)
    if (encoding[i] > 1) encoding[i] = 3;
}

__global__
void firstFiltering(int *g_nodesCount,
                    int *g_edgesCount,
                    int *g_encoding,
                    int *q_nodesCount,
                    int *q_edgesCount,
                    int *q_encoding,
                    int *q_candidates) {
  int node_q = blockIdx.x * blockDim.x + threadIdx.x; // thread_id
  if (node_q >= q_nodesCount[0]) return;
  int start_q = g_edgesCount[0] * node_q;
//  int end_q = start_q + q->edgesCount; // useless
  for (VID_t node_g = 0; node_g < g_nodesCount[0]; ++node_g) {
    bool flag = true;
    int start_g = g_edgesCount[0] * node_g;
//    int end_g = start_g + g->edgesCount; // useless
    for (int i = 0; i < q_edgesCount[0]; ++i) {
      if ((q_encoding[start_q + i] & g_encoding[start_g + i]) != q_encoding[start_q + i]) {
        flag = false;
        break;
      }
    }
    if (flag) q_candidates[node_q * g_nodesCount[0] + node_g] = 1;
  }
}

//__global__
//void secondFiltering(Graph *g, Graph *q){
//  int node_q = blockIdx.x * blockDim.x + threadIdx.x; // thread_id
//  if (node_q >= q->nodesCount) return;
//}

__global__
void FindCandidateEdges(Graph *g, Graph *q) {
  int node_q = blockIdx.x * blockDim.x + threadIdx.x; // thread_id
  if (node_q >= q->nodesCount) return;

}