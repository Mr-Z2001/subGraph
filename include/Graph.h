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


class Graph {
public:
  int partitionSize;
  int nodesCount;
  int edgesCount;

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
  double *score;
  int *order;

  int *inDegree;
  int *outDegree;

  VID_t* core;
  VID_t* forest;

public:
  void load(const char *filename, std::vector<std::vector<int>> &adj);

  void construct(std::vector<std::vector<int>> &adj) const;

  void deleteNode(VID_t v);

  void decreaseDegree(VID_t v) const;

  [[nodiscard]] int getDegree(VID_t v) const;

  [[nodiscard]] int getInDegree(VID_t v) const;

  [[nodiscard]] int getOutDegree(VID_t v) const;

};


#endif