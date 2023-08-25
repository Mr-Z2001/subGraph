#ifndef SUBGRAPHISOMORPHISM_FILTER_H
#define SUBGRAPHISOMORPHISM_FILTER_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Graph.h"

__global__ void getEncoding(const Graph *g);

__global__ void firstFiltering(Graph *g, Graph *q);

#endif //SUBGRAPHISOMORPHISM_FILTER_H
