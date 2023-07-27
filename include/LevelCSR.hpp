//
// Created by z on 23-7-3.
//

#ifndef SUBGRAPHISOMORPHISM_LEVELCSR_H
#define SUBGRAPHISOMORPHISM_LEVELCSR_H

#include <vector>

#include "MatrixElement.h"

template<class nodeValue_t, class edgeValue_t>
class LevelCSR {
private:
  std::vector<nodeValue_t> nodes;
  std::vector<int> index1; // index to index2
  std::vector<int> index2; // index to col, index2[i] denotes the first element in the partition.
  std::vector<int> col; // col[i] denotes the column of the i-th element.
  std::vector<edgeValue_t> edgeValues;
  int partitionSize;
public:
  LevelCSR() = default;

  ~LevelCSR() = default;

  void setPartitionSize(int _partitionSize) {
    LevelCSR::partitionSize = _partitionSize;
  }

  void construct(std::vector<MatrixElement<edgeValue_t>> vme, int nodeCount) {
    nodes.resize(nodeCount);
    index1.resize(nodeCount + 1);
    index2.resize(nodeCount + 1);
    col.resize(vme.size());
    edgeValues.resize(vme.size());
    for (int i = 0; i < vme.size(); ++i) {
      col[i] = vme[i].getCoordinate().getCol();
      edgeValues[i] = vme[i].getEdgeValue();
    }
    int currentRow = -1;
    int currentPartition = -1;
    int currentPartitionSize = 0;
    int elementRow, elementCol;
    for (int i = 0; i < vme.size(); ++i) {
      elementRow = vme[i].getCoordinate().getRow();
      elementCol = vme[i].getCoordinate().getCol();
      col[i] = elementCol;
      edgeValues[i] = vme[i].getEdgeValue();
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
};

#endif //SUBGRAPHISOMORPHISM_LEVELCSR_H
