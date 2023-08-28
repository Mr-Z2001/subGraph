//
// Created by z on 23-8-22.
//

#include "Graph.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef SUBGRAPHISOMORPHISM_DECOMPOSITION_H
#define SUBGRAPHISOMORPHISM_DECOMPOSITION_H

void getCoreForest(Graph g, int *core, int *forest);

__global__ void
getScore(int *nodesCount, int *candidateSize, int *candidates, int *inDegree, int *outDegree, float *score);

#endif //SUBGRAPHISOMORPHISM_DECOMPOSITION_H
