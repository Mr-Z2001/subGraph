#include "Graph.h"
#include "filter.cu"
#include "GraphElement.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

std::vector<Node_t> check(LevelCSR &graph) {
  std::vector<Node_t> ret;
  for (auto node: graph.getNodesVec())
    if (graph.getDegree(node.getId()) == 1)
      ret.push_back(node.getId()), graph.deleteNode(node.getId());
  return ret;
}

void coreForestDecomposition(LevelCSR &graph,
                             LevelCSR &core, LevelCSR &forest) {
  std::vector<Node_t> oneDegreeNodes;

  while(true){
    oneDegreeNodes.clear();
    for (auto node: graph.getNodesVec())
      if (graph.getDegree(node.getId()) == 1)
        oneDegreeNodes.push_back(node.getId()), graph.deleteNode(node.getId());
    if (oneDegreeNodes.empty()) break;
  }

  for (auto node: graph.getNodesVec()) {
    if (graph.getDegree(node.getId()) == 0) continue;
    if (graph.getDegree(node.getId()) == 1) {
      oneDegreeNodes.push_back(node.getId());
      graph.deleteNode(node.getId());
    }
  }
}

void Join(LevelCSR query, LevelCSR graph, std::vector<std::vector<Node>> candidates,
          LevelCSR *result) {
  LevelCSR P, Qtemp = query;
  std::vector<Node_t> core, forest;
  std::vector<std::pair<Node_t, int>> score;
  std::vector<Node_t> sequence;
  std::vector<std::vector<Edge>> M;

  score.resize(query.getNodesVec().size() + 1);
  while (true) {
    std::vector<Node_t> f;
    if (!((f = check(Qtemp)).empty())) {
      forest.push_back(f);
    } else {
      core = Qtemp;
      break;
    }
  }

  for (auto node: core.getNodesVec()) {
    score[node.getId()] = {node.getId(), candidates[node.getId()].size() / core.getDegree(node.getId())};
  }
  std::sort(score.begin(), score.end(), [](auto a, auto b) { return a.second > b.second; });
  for (auto s: score) {
    if (s.second) sequence.push_back(s.first);
    else break;
  }

  for (auto nodes: forest) {
    score.clear();
    for (auto node: nodes) {
      score[node] = {node, candidates[node].size()};
    }
    std::sort(score.begin(), score.end(), [](auto a, auto b) { return a.second > b.second; });
    for (auto s: score) {
      sequence.push_back(s.first);
    }
  }



  for ( auto node : sequence){

  }
}