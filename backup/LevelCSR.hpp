#ifndef SUBGRAPHISOMORPHISM_LEVELCSR_H
#define SUBGRAPHISOMORPHISM_LEVELCSR_H

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "MatrixElement.h"
#include "GraphElement.hpp"

class LevelCSR {
private:
  std::vector<Node> nodes;
  std::vector<int> index1; // index to index2
  std::vector<int> index2; // index to col, index2[i] denotes the first element in the partition.
  std::vector<int> col; // col[i] denotes the column of the i-th element.
  std::vector<Edge> edges;
  int partitionSize;
public:
  LevelCSR() = default;

  ~LevelCSR() = default;

  void setPartitionSize(int _partitionSize) {
    LevelCSR::partitionSize = _partitionSize;
  }

  void construct(std::vector<MatrixElement> vme, int nodeCount) {
    nodes.resize(nodeCount);
    index1.resize(nodeCount + 1);
    index2.resize(nodeCount + 1);
    col.resize(vme.size());
    edges.resize(vme.size());
    for (int i = 0; i < vme.size(); ++i) {
      col[i] = vme[i].getCoordinate().getCol();
      edges[i] = vme[i].getEdgeValue();
//      edges[i] = vme[i].getEdgeValue();
    }
    int currentRow = -1;
    int currentPartition = -1;
    int currentPartitionSize = 0;
    int elementRow, elementCol;
    for (int i = 0; i < vme.size(); ++i) {
      elementRow = vme[i].getCoordinate().getRow();
      elementCol = vme[i].getCoordinate().getCol();
      col[i] = elementCol;
      edges[i] = vme[i].getEdgeValue();
      if (elementRow != currentRow) {
        currentPartition++;
        currentRow = elementRow;
        index1[currentRow] = currentPartition;
        index2[currentPartition] = i;
        currentPartitionSize = 1;
      } else {
        currentPartitionSize++;
        if (currentPartitionSize == partitionSize + 1) {
          currentPartition++;
          index2[currentPartition] = i + 1;
          currentPartitionSize = 1;
        }
      }
    }
  }

  void deleteNode(Node_t node) {
    Node_t u;
    for (int i = 0; i < nodes.size(); ++i) {
      if (nodes[i].getId() == node) {
        u = i;
        break;
      }
    }
    int start = index2[index1[u]];
    int end = index2[index1[u] + 1];
    int counter = end - start;
    edges.erase(edges.begin() + start, edges.begin() + end);
    col.erase(col.begin() + start, col.begin() + end);
    for (int i = index1[u] + 1; i < index1.size(); ++i) {
      index2[i] -= counter;
    }
    index2.erase(index2.begin() + index1[u], index2.begin() + index1[u] + 1);
    for (int i = index1.size() - 1; i > index1[u]; --i) {
      index1[i - 1] = index1[i];
    }
    index1.pop_back();
    nodes.erase(nodes.begin() + u, nodes.begin() + u + 1);
  }

  // TODO: review this function
  void addEdge(Edge e) {
    bool isNewSrc = true;
    Node_t src = e.getSrc();
    Node_t dest = e.getDst();
    int pos = 0;
    for (auto node: nodes) {
      if (node.getId() == src) {
        isNewSrc = false;
        break;
      } else pos++;
    }
    if (isNewSrc) {
      Node n(src, -1);
      nodes.push_back(n);
      index1.push_back(index2.size());
      index2.push_back(col.size());
      col.push_back(dest);
      edges.push_back(e);
    } else { // existing src node
      int end = index2[index1[pos] + 1];
      int end_1 = index2[index1[pos]];
      if (end - end_1 == partitionSize) { // partition is full
        index2.insert(index2.begin() + index1[pos + 1], end);
        col.insert(col.begin() + end, dest);
        edges.insert(edges.begin() + end, e);
        for (int i = index1[pos + 1]; i < index1.size(); ++i) index2[i]++;
      } else { // partition is not full
        int start = index2[index1[pos]];
        col.insert(col.begin() + start, dest);
        edges.insert(edges.begin() + start, e);
        for (int i = index1[pos] + 1; i < index1.size(); ++i) index2[i]++;
      }
    }
  }

  int getDegree(Node_t u) {
    return index2[index1[u + 1]] - index2[index1[u]];
  }

  [[nodiscard]] std::vector<Edge> getEdgesVec() const {
    return edges;
  }

  [[nodiscard]] std::vector<int> getColVec() const {
    return col;
  }

  [[nodiscard]] std::vector<int> getIndex1Vec() const {
    return index1;
  }

  [[nodiscard]] std::vector<int> getIndex2Vec() const {
    return index2;
  }

  [[nodiscard]] std::vector<Node> getNodesVec() const {
    return nodes;
  }

  [[nodiscard]] int getPartitionSize() const {
    return partitionSize;
  }

  Edge *getEdgesData() {
    return edges.data();
  }

  int *getColData() {
    return col.data();
  }

  int *getIndex1Data() {
    return index1.data();
  }

  int *getIndex2Data() {
    return index2.data();
  }

  Node *getNodesData() {
    return nodes.data();
  }

  Node getNode(int i) {
    if (i > nodes.size() - 1) return nodes[nodes.size() - 1];
    return nodes[i];
  }

  int getIndex1(int i) {
    if (i > index1.size() - 1) return index1[index1.size() - 1];
    return index1[i];
  }

  int getIndex2(int i) {
    if (i > index2.size() - 1) return index2[index2.size() - 1];
    return index2[i];
  }

  int getCol(int i) {
    if (i > col.size() - 1) return col[col.size() - 1];
    return col[i];
  }

  Edge getEdges(int i) {
    if (i > edges.size() - 1) return edges[edges.size() - 1];
    return edges[i];
  }
};

#endif //SUBGRAPHISOMORPHISM_LEVELCSR_H
