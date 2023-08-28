//
// Created by z on 23-8-20.
//

#ifndef SUBGRAPHISOMORPHISM_LEVELCSR_H
#define SUBGRAPHISOMORPHISM_LEVELCSR_H

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "MatrixElement.h"
#include "GraphElement.hpp"

typedef int VID_t;
typedef int EID_t;
typedef int Label_t;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char *const func, const char *const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


class Graph {
public:
  int partitionSize;
  int nodesCount;
  int edgesCount;
  int edgeLabelsCount;
  int idx2Size;
  int encodingSize;

  VID_t *nodes; // CSR
  int *row; // CSR
  int *column; // CSR
  Label_t *edges_labels; // CSR
  Label_t *node_labels; // CSR

  int *index1; // Level-CSR
  int *index2; // Level-CSR

  int *encoding;
  int *candidates;
  int candidateSize;
  float *score;
  int *order;

  int *inDegree;
  int *outDegree;

  VID_t *core;
  VID_t *forest;

public:
  void load(const char *filename, std::vector<std::vector<std::pair<VID_t, Label_t>>> &adj);

  void construct(std::vector<std::vector<std::pair<VID_t, Label_t>>> &adj) const;

  void deleteNode(VID_t v);

  void decreaseDegree(VID_t v, bool* deleted) const;

  [[nodiscard]] int getDegree(VID_t v) const;

  [[nodiscard]] int getInDegree(VID_t v) const;

  [[nodiscard]] int getOutDegree(VID_t v) const;

};


#endif