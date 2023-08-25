#include <fstream>
#include <cinttypes>
#include "Graph.h"
#include <algorithm>

#include <thrust/host_vector.h>

//#define DEBUG
#define DEBUG_C

/**
 * n, m denotes the number of vertices and edges respectively.
 * following n lines, each line contains a vertex id and a vertex label.
 * following m lines, each line contains a source vertex id, a target vertex id, and an edge label.
 */
void Graph::load(const char *filename, std::vector<std::vector<int>> &adj) {
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
  n++, m++;
  nodes = (VID_t *) realloc(nodes, n * sizeof(VID_t));
  row = (int *) realloc(row, n * sizeof(int));
  inDegree = (int *) realloc(inDegree, n * sizeof(int));
  outDegree = (int *) realloc(outDegree, n * sizeof(int));
  index1 = (int *) realloc(index1, n * sizeof(int));
  core = (VID_t *) realloc(core, n * sizeof(VID_t));
  forest = (VID_t *) realloc(forest, n * sizeof(VID_t));
  score = (double *) realloc(score, n * sizeof(double));
  order = (int *) realloc(order, n * sizeof(int));
  index2 = (int *) realloc(index2, m * sizeof(int));
  node_labels = (Label_t *) realloc(node_labels, n * sizeof(Label_t));
  edges_labels = (EID_t *) realloc(edges_labels, m * sizeof(EID_t));
  column = (int *) realloc(column, m * sizeof(int));
//  encoding = (int *) realloc(encoding, n * m * sizeof(int));
  adj.resize(n);

#ifdef DEBUG //passed
  std::cout << "nodesCount: " << nodesCount << std::endl;
  std::cout << "edgesCount: " << edgesCount << std::endl;
#endif


//  thrust::host_vector<thrust::host_vector<int>> adj(n);
  int id, label, from, to;
  for (int i = 0; i < n; ++i) {
    f >> id >> label;
    nodes[i] = id; // `VID` asc required.
    node_labels[i] = label;
    adj[i].resize(0);
  }

#ifdef DEBUG // passed
  std::cout << "nodes loaded" << std::endl;
#endif

  for (int i = 0; i < m; ++i) {
    f >> from >> to >> label; // `from` asc required.
#ifdef DEBUG
    std::cout << from << ' ' << to << ' ' << label << std::endl;
#endif
    adj[from].push_back(to);
    inDegree[to]++;
    outDegree[from]++;
    edges_labels[i] = label;
    column[i] = to;
  }
#ifdef DEBUG // passed
  std::cout << "edges loaded" << std::endl;
#endif
  f.close();
}

// make CSR to Level-CSR, i.e. fill index1 and index2.
void Graph::construct(std::vector<std::vector<int>> &adj) const {
//  int n = nodesCount, m = edgesCount;
  int part_count, remain;
  int idx1_ptr = 0, idx2_ptr = 0, col_ptr = 0;
  for (int i = 0; i < nodesCount; ++i) {
    int neighbor_count = adj[i].size();
    if (!neighbor_count) continue;
    part_count = neighbor_count / partitionSize;
    remain = neighbor_count % partitionSize;

#ifdef DEBUG_C // passed
    std::cout << "i: " << i << std::endl;
    std::cout << "part_count: " << part_count << std::endl;
    std::cout << "remain: " << remain << std::endl;
#endif
    for (int ii = 0; ii < part_count; ++ii) {
      index2[idx2_ptr + ii] = col_ptr + ii * partitionSize;
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

void Graph::decreaseDegree(VID_t v) const {
  inDegree[v] = 0;
  outDegree[v] = 0;
  for (int i = 0; i < nodesCount; ++i) {
    if (i == v) continue;
    for (int j = index2[index1[i]]; j < index2[index1[i + 1]]; ++j) {
      if (column[j] == v) {
        inDegree[i]--;
        outDegree[i]--;
        break;
      }
    }
  }
//  for (int to = index2[index1[v]]; to < index2[index1[v + 1]]; ++to) {
//    inDegree[column[to]]--;
//    outDegree[column[to]]--;
//  }
}