
#ifndef SUBGRAPHISOMORPHISM_JOIN_H
#define SUBGRAPHISOMORPHISM_JOIN_H

#include "Graph.h"

void join(Graph* q, Graph* g, int** M);
bool join(Graph* q, Graph* g, int* M, VID_t node_q);

#endif //SUBGRAPHISOMORPHISM_JOIN_H
