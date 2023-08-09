//
// Created by z on 23-7-3.
//

#ifndef SUBGRAPHISOMORPHISM_LEVELCSR_H
#define SUBGRAPHISOMORPHISM_LEVELCSR_H

#include <vector>

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
    if(i > nodes.size() - 1) return nodes[nodes.size() - 1];
    return nodes[i];
  }

  int getIndex1(int i) {
    if(i > index1.size() - 1) return index1[index1.size() - 1];
    return index1[i];
  }

  int getIndex2(int i) {
    if(i > index2.size() - 1) return index2[index2.size() - 1];
    return index2[i];
  }

  int getCol(int i) {
    if(i > col.size() - 1) return col[col.size() - 1];
    return col[i];
  }

  Edge getEdges(int i) {
    if(i > edges.size() - 1) return edges[edges.size() - 1];
    return edges[i];
  }
};

#endif //SUBGRAPHISOMORPHISM_LEVELCSR_H
