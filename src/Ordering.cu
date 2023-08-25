#include "Ordering.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define DEBUG

void getCoreForest(Graph g, int *core, int *forest) {
  std::vector<VID_t> oneDegreeNodes;

  while (true) {
    oneDegreeNodes.clear();
    for (int node_g = 0; node_g < g.nodesCount; ++node_g) {
      if (g.getDegree(node_g) == 1) {
//        oneDegreeNodes.push_back(node_g);
//        g.deleteNode(node_g);
#ifdef DEBUG
        std::cout << "node_g: " << node_g << std::endl;
#endif
        g.decreaseDegree(node_g);
      }
    }
    if (oneDegreeNodes.empty()) break;
  }
#ifdef DEBUG
  std::cout << "oneDegreeNodes done." << std::endl;
#endif

  for (int node_g = 0; node_g < g.nodesCount; ++node_g) {
    if (g.getDegree(node_g) == 0) forest[node_g] = 1;
    else core[node_g] = 1;
  }
#ifdef DEBUG
  std::cout << "core-forest done." << std::endl;
//  for(int i = 0; i < g.nodesCount; ++i) std::cout << core[i] << ' ';
//  std::cout << std::endl;
//  for(int i = 0; i < g.nodesCount; ++i) std::cout << forest[i] << ' ';
//  std::cout << std::endl;

#endif
}

__global__ void getScore(Graph *g) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= g->nodesCount) return;
  int cnt = 0;
  for (int i = tid * g->candidateSize; i < (tid + 1) * g->candidateSize; ++i)
    if (g->candidates[i] == 1) cnt++;
  g->score[tid] = cnt * 1.0 / (g->inDegree[tid] + g->outDegree[tid]);
}