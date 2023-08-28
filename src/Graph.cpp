#include <fstream>
#include <cinttypes>
#include "Graph.h"
#include <algorithm>
#include <set>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#define DEBUG
//#define DEBUG_C

/**
 * n, m denotes the number of vertices and edges respectively.
 * following n lines, each line contains a vertex id and a vertex label.
 * following m lines, each line contains a source vertex id, a target vertex id, and an edge label.
 */
void Graph::load(const char *filename, std::vector<std::vector<std::pair<VID_t, Label_t>>> &adj) {
  std::fstream f;
  f.open(filename, std::ios::in);
  if (!f.is_open()) {
    std::cerr << "Error opening file " << filename << std::endl;
    exit(1);
  } else {
    std::cout << "Loading file " << filename << std::endl;
  }
  int64_t n, m;
  f >> n >> m;
#ifdef DEBUG //passed
  std::cout << n << ' ' << m << std::endl;
#endif
  nodesCount = n;
  edgesCount = m;
  n++, m <<= 1;
  m++;
  nodes = (VID_t *) realloc(nodes, n * sizeof(VID_t));
  row = (int *) realloc(row, n * sizeof(int));
  inDegree = (int *) realloc(inDegree, n * sizeof(int));
  outDegree = (int *) realloc(outDegree, n * sizeof(int));
  index1 = (int *) realloc(index1, n * sizeof(int));
  core = (VID_t *) realloc(core, n * sizeof(VID_t));
  forest = (VID_t *) realloc(forest, n * sizeof(VID_t));
  score = (float *) realloc(score, n * sizeof(float));
  order = (int *) realloc(order, n * sizeof(int));
  index2 = (int *) realloc(index2, m * sizeof(int));
  node_labels = (Label_t *) realloc(node_labels, n * sizeof(Label_t));
  edges_labels = (EID_t *) realloc(edges_labels, m * sizeof(EID_t));
  column = (int *) realloc(column, m * sizeof(int));
//  encoding = (int *) realloc(encoding, n * m * sizeof(int));
  adj.resize(n);

  memset(nodes, 0, n * sizeof(VID_t));
  memset(row, 0, n * sizeof(int));
  memset(inDegree, 0, n * sizeof(int));
  memset(outDegree, 0, n * sizeof(int));
  memset(index1, -1, n * sizeof(int));
  memset(core, 0, n * sizeof(VID_t));
  memset(forest, 0, n * sizeof(VID_t));
  memset(score, 0, n * sizeof(double));
  memset(order, -1, n * sizeof(int));
  memset(index2, 0, m * sizeof(int));
  memset(node_labels, -1, n * sizeof(Label_t));
  memset(edges_labels, -1, m * sizeof(EID_t));
  memset(column, -1, m * sizeof(int));

#ifdef DEBUG //passed
  std::cout << "nodesCount: " << nodesCount << std::endl;
  std::cout << "edgesCount: " << edgesCount << std::endl;
#endif


//  thrust::host_vector<thrust::host_vector<int>> adj(n);
  int id, label, from, to;
  for (int i = 0; i < nodesCount; ++i) {
    f >> id >> label;
    std::cout << id << ' ' << label << std::endl;
    nodes[i] = id; // `VID` asc required.
    node_labels[i] = label;
    adj[i].resize(0);
  }

#ifdef DEBUG // passed
  std::cout << "nodes loaded" << std::endl;
#endif

  std::set<int> labels;
  for (int i = 0; i < edgesCount; ++i) {
    f >> from >> to >> label;
#ifdef DEBUG
    std::cout << from << ' ' << to << ' ' << label << std::endl;
#endif
    // bidirectional
    adj[from].push_back({to, label});
    adj[to].push_back({from, label});
    inDegree[to]++, inDegree[from]++;
    outDegree[from]++, outDegree[to]++;
//    edges_labels[i] = label;
//    column[i] = to;
    labels.insert(label);
  }
  // edge labels & column
  int idx = 0;
  for (int i = 0; i < nodesCount; ++i) {
    std::sort(adj[i].begin(), adj[i].end());
    for (auto &j: adj[i]) {
      edges_labels[idx] = j.second;
      column[idx] = j.first;
      idx++;
    }
  }

#ifdef DEBUG // passed
  std::cout << "edges loaded" << std::endl;
#endif
  f.close();
  this->edgeLabelsCount = labels.size();
  edgesCount <<= 1;
}

// make CSR to Level-CSR, i.e. fill index1 and index2.
void Graph::construct(std::vector<std::vector<std::pair<VID_t, Label_t>>> &adj) const {
//  int n = nodesCount, m = edgesCount;
  int part_count, remain;
  int idx1_ptr = 0, idx2_ptr = 0, col_ptr = 0;
  for (int i = 0; i < nodesCount; ++i) {
    int neighbor_count = adj[i].size();
    std::cout << "neighbor_count = " << neighbor_count << std::endl;
    if (!neighbor_count) continue;
    part_count = neighbor_count / partitionSize;
    remain = neighbor_count % partitionSize;

#ifdef DEBUG_C // passed
    std::cout << "i: " << i << std::endl;
    std::cout << "part_count: " << part_count << std::endl;
    std::cout << "remain: " << remain << std::endl;
#endif
    for (int part_idx = 0; part_idx < part_count + std::min(remain, 1); ++part_idx) {
      index2[idx2_ptr + part_idx] = col_ptr + part_idx * partitionSize;
    }
    index1[idx1_ptr] = idx2_ptr;

    col_ptr += neighbor_count;
    idx2_ptr += part_count + std::min(remain, 1);
    idx1_ptr += 1;
  }

  index1[idx1_ptr] = idx2_ptr;
  index2[idx2_ptr] = col_ptr;
}

/**
 * Get the out degree of vertex `v`.
 * @param v
 * @return degree Value
 */
int Graph::getDegree(VID_t v) const {
  return inDegree[v] + outDegree[v];
}

int Graph::getInDegree(VID_t v) const {
  return inDegree[v];
}

int Graph::getOutDegree(VID_t v) const {
  return outDegree[v];
}

void Graph::deleteNode(VID_t v) {

}

void Graph::decreaseDegree(VID_t v, bool* deleted) const {
  inDegree[v] = 0;
  outDegree[v] = 0;
  deleted[v] = true;
  int neighbor_count = index2[index1[v + 1]] - index2[index1[v]];
  for (int neighbor = index2[index1[v]]; neighbor < index2[index1[v + 1]]; ++neighbor) {
    if (column[neighbor] == v) continue;
    if (deleted[column[neighbor]]) continue;
    inDegree[column[neighbor]]--;
    outDegree[column[neighbor]]--;
  }

//  for (int to = index2[index1[v]]; to < index2[index1[v + 1]]; ++to) {
//    inDegree[column[to]]--;
//    outDegree[column[to]]--;
//  }
}