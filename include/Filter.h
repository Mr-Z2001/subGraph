#ifndef SUBGRAPHISOMORPHISM_FILTER_H
#define SUBGRAPHISOMORPHISM_FILTER_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Graph.h"

__global__
void getEncoding(int *nodesCount,
                 int *edgesCount,
                 int *edges_labels,
                 int *index1,
                 int *index2,
                 int *encoding);

__global__
void firstFiltering(int *g_nodesCount,
                    int *g_edgesCount,
                    int *g_encoding,
                    int *q_nodesCount,
                    int *q_edgesCount,
                    int *q_encoding,
                    int *q_candidates);

#endif //SUBGRAPHISOMORPHISM_FILTER_H
