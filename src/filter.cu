//
// Created by z on 23-7-27.
//

#include "LevelCSR.hpp"
#include "GraphElement.hpp"

#include <vector>
#include <string>
#include <bitset>
#include <set>
#include <algorithm>

void getSig(const LevelCSR &graph, std::vector<int> &sigs) {
  std::set<int> sigSet;

  auto edges = graph.getEdgesVec();
  for (const auto &edge: edges) {
    const auto label = edge.getLabel();
    sigSet.insert(label);
  }

  for (const auto &label: sigSet)
    sigs.push_back(label);
  std::sort(sigs.begin(), sigs.end());
}

//__global__
//void getSigVec_kernel(LevelCSR graph, int** sigs, Node nodes[], int nodecnt, int sig[]) {
//  int tid = threadIdx.x + blockIdx.x * blockDim.x;
//  if (tid < nodecnt) {
//    Node_t node = nodes[tid].getId();
//    int start = graph.getCol(graph.getIndex2(graph.getIndex1(node)));
//    int end = graph.getCol(graph.getIndex2(graph.getIndex1(node + 1)));
////    int start = graph.getColData()[graph.getIndex2Data()[graph.getIndex1Data()[node]]];
////    int end = graph.getColData()[graph.getIndex2Data()[graph.getIndex1Data()[node] + 1]];
//    for (int i = start; i < end; ++i) {
//      Edge_t edge = graph.getEdges(i).getLabel();
//      sigs[tid][edge]++;
//    }
//  }
//}

void getSigVec(LevelCSR &graph, std::vector<std::vector<int>> &sig) {
  auto nodes = graph.getNodesVec();
  for (const auto &node: nodes) {
    Node_t id = node.getId();
    int start = graph.getCol(graph.getIndex2(graph.getIndex1(id)));
    int end = graph.getCol(graph.getIndex2(graph.getIndex1(id + 1)));
    for (int i = start; i < end; ++i) {
      Edge_t edge = graph.getEdges(i).getLabel();
      sig[id][edge]++;
      if (sig[id][edge] > 1) sig[id][edge] = 2;
    }
  }
}


std::vector<std::vector<Node>> filterCandidates(LevelCSR &query,
                                                LevelCSR &graph) {
  std::vector<std::vector<Node>> candidates;

// encoding to S(u) and S(v)
  std::vector<int> querySigs;
  std::vector<int> graphSigs;
  getSig(query, querySigs);
  getSig(graph, graphSigs);
  std::vector<std::vector<int>> querySigVec;
  std::vector<std::vector<int>> graphSigVec;
  getSigVec(query, querySigVec);
  getSigVec(graph, graphSigVec);

  for (auto &u: query.getNodesVec()) {
    for (auto &v: graph.getNodesVec()) {
      if (querySigVec[u.getId()] == graphSigVec[v.getId()]) {
        candidates[u.getId()].push_back(v);
      }
    }
  }
  for (auto &u: query.getNodesVec()) {
    for (auto &v: candidates[u.getId()]) {
      auto deg_v = graph.getDegree(v.getId());
      if (deg_v < 32 or deg_v > 512) continue;

      // check if the u's and v's first neighbors are match
      std::vector<std::pair<Edge_t, Node_t>> u_neighbors, v_neighbors;
      for (auto &edge: query.getEdgesVec()) {
        if (edge.getSrc() == u.getId()) {
          u_neighbors.emplace_back(edge.getLabel(), edge.getDst());
        }
      }
      for (auto &edge: graph.getEdgesVec()) {
        if (edge.getSrc() == v.getId()) {
          v_neighbors.emplace_back(edge.getLabel(), edge.getDst());
        }
      }
      std::sort(u_neighbors.begin(), u_neighbors.end());
      std::sort(v_neighbors.begin(), v_neighbors.end());

      // two-pointers, check if u_neighbors is v_neighbors subset
      int i = 0, j = 0;
      while (i < u_neighbors.size() and j < v_neighbors.size()) {
        if (u_neighbors[i].first == v_neighbors[j].first) {
          if (u_neighbors[i].second != v_neighbors[j].second) break;
          ++i;
          ++j;
        } else if (u_neighbors[i].first < v_neighbors[j].first) {
          ++i;
        } else {
          ++j;
        }
      }
      if (i == u_neighbors.size()) {
        candidates[u.getId()].push_back(v);
      }
    }
  }
  return candidates;
}








































