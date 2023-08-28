#include "Join.h"

//void join(Graph *q, Graph *g, int **M) {
//  for (VID_t node_q = 0; node_q < q->nodesCount; ++node_q) {
//    for (VID_t can = 0; can < q->candidateSize; ++can) {
//      if (q->candidates[node_q * q->candidateSize + can] == 0) continue;
//
//    }
//  }
//}

bool join(Graph *q, Graph *g, int *M, VID_t node_q) {
  std::cout << q->nodesCount << std::endl;
  if (node_q == q->nodesCount) return true;

  std::cout << "Entering " << node_q << std::endl;

  for (VID_t can = 0; can < q->candidateSize; ++can) {
    std::cout << "can = " << can << std::endl;
    if (q->candidates[node_q * q->candidateSize + can] == -1) continue;
    bool flag = true;
    for (VID_t node_p = 0; node_p < node_q; ++node_p) {
      if (q->candidates[node_p * q->candidateSize + can] == -1) continue;
      if (q->edges_labels[node_q * q->nodesCount + node_p] != g->edges_labels[M[node_q] * g->nodesCount + M[node_p]]) {
        flag = false;
        break;
      }
    }
    if (flag) {
      M[node_q] = can;
      if (join(q, g, M, node_q + 1)) return true;
    }
  }
  return false;
}