//
// Created by z on 23-8-21.
//
#include "Filter.h"
#include "Graph.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void getEncoding(const Graph *g) {
  int node_id = blockIdx.x * blockDim.x + threadIdx.x; // thread_id
  if (node_id >= g->nodesCount) return;
  int start = g->index2[g->index1[node_id]];
  int end = g->index2[g->index1[node_id + 1]];
  Label_t l;
  for (int i = start; i < end; ++i) {
    l = g->edges_labels[i];
    g->encoding[node_id * g->edgesCount + l] += 1;
  }
//  cudaThreadSynchronize();
//  cudaDeviceSynchronize();
  __syncthreads();
  // encode to 00, 01, 11
  // stable complexity: O(m) per thread. Avoiding too slow in case of high degree.
  for (int i = node_id * g->edgesCount; i < (node_id + 1) * g->edgesCount; ++i)
    if (g->encoding[i] > 1) g->encoding[i] = 3;
}

__global__
void firstFiltering(Graph *g, Graph *q) {
  int node_q = blockIdx.x * blockDim.x + threadIdx.x; // thread_id
  if (node_q >= q->nodesCount) return;
  int start_q = g->edgesCount * node_q;
//  int end_q = start_q + q->edgesCount; // useless
  for (VID_t node_g = 0; node_g < g->nodesCount; ++node_g) {
    bool flag = true;
    int start_g = g->edgesCount * node_g;
//    int end_g = start_g + g->edgesCount; // useless
    for (int i = 0; i < q->edgesCount; ++i) {
      if ((q->encoding[start_q + i] & g->encoding[start_g + i]) != q->encoding[start_q + i]) {
        flag = false;
        break;
      }
    }
    if (flag) g->candidates[node_q * g->nodesCount + node_g] = 1;
  }
}

//__global__
//void secondFiltering(Graph *g, Graph *q){
//  int node_q = blockIdx.x * blockDim.x + threadIdx.x; // thread_id
//  if (node_q >= q->nodesCount) return;
//}

