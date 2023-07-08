//
// Created by z on 23-7-3.
//

#ifndef SUBGRAPHISOMORPHISM_LEVELCSR_H
#define SUBGRAPHISOMORPHISM_LEVELCSR_H

#include <vector>

typedef std::pair<int, int> pii;

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

    void addNode(nodeValue_t node, int _row, int _col){

    }



};

#endif //SUBGRAPHISOMORPHISM_LEVELCSR_H
