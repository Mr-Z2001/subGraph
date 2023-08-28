#include "Ordering.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define DEBUG

void getCoreForest(Graph g, int *core, int *forest) {
  std::vector<VID_t> oneDegreeNodes;
  bool *deleted = new bool[g.nodesCount];
  memset(deleted, false, g.nodesCount * sizeof(bool));

  while (true) {
    oneDegreeNodes.clear();
    for (int node_g = 0; node_g < g.nodesCount; ++node_g) {
      std::cout << "node_g: " << node_g << " degree: " << g.getDegree(node_g) << std::endl;
      if (g.getDegree(node_g) == 2) {
        oneDegreeNodes.push_back(node_g);
//        g.deleteNode(node_g);
#ifdef DEBUG
        std::cout << "node_g: " << node_g << std::endl;
#endif
        g.decreaseDegree(node_g, deleted);
      }
    }
    if (oneDegreeNodes.empty()) break;
  }
#ifdef DEBUG
  std::cout << "oneDegreeNodes done." << std::endl;
#endif

  for (int node_g = 0; node_g < g.nodesCount; ++node_g) {
    if (!deleted[node_g]) core[node_g] = 1;
    else forest[node_g] = 1;
  }
#ifdef DEBUG
  std::cout << "core-forest done." << std::endl;
//  for(int i = 0; i < g.nodesCount; ++i) std::cout << core[i] << ' ';
//  std::cout << std::endl;
//  for(int i = 0; i < g.nodesCount; ++i) std::cout << forest[i] << ' ';
//  std::cout << std::endl;

#endif
}

__global__ void
getScore(int *nodesCount, int *candidateSize, int *candidates, int *inDegree, int *outDegree, float *score) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nodesCount[0]) return;
  int cnt = 0;
  for (int i = tid * candidateSize[0]; i < (tid + 1) * candidateSize[0]; ++i)
    if (candidates[i] == 1) cnt++;
  printf("node_id: %d, cnt: %d\n", tid, cnt);
  score[tid] = cnt * 1.0 / inDegree[tid];
}