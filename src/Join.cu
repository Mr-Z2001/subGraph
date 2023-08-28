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
//  std::cout << q->nodesCount << std::endl;
  if (node_q == q->nodesCount) return true;

  std::cout << "Entering " << node_q << std::endl;

  for (VID_t can = 0; can < q->candidateSize; ++can) {
//    std::cout << "can = " << can << std::endl;
    if (q->candidates[node_q * q->candidateSize + can] == 0) continue; // `can` is not a candidate of `node_q`
    bool flag = true;

    // walking through all the neighbors of `node_q`
    int start = q->index2[q->index1[node_q]];
    int end = q->index2[q->index1[node_q + 1]];
    for (int i = start; i < end; ++i) {
      VID_t q_nb = q->column[i], can_nb;
      Label_t q_label = q->edges_labels[i], g_label;
      if (M[q_nb] == -1) continue; // `nb` is not mapped to any node in `g`
      // check if edge_label is the same
      can_nb = M[q_nb];
      for(int j = g->index2[g->index1[can]]; j < g->index2[g->index1[can + 1]]; ++j) {
        if (g->column[j] == can_nb) {
          g_label = g->edges_labels[j];
          break;
        }
      }
      if (q_label != g_label) {
        flag = false;
        break;
      }
    }
    if (flag) {
      M[node_q] = can;
      if (join(q, g, M, node_q + 1)) return true;
      M[node_q] = -1;
    }
  }
  return false;
}