//
// Created by z on 23-7-23.
//

#ifndef SUBGRAPHISOMORPHISM_MATRIXELEMENT_H
#define SUBGRAPHISOMORPHISM_MATRIXELEMENT_H

#include "Coordinate.h"

#include "GraphElement.hpp"

class MatrixElement {
private:
  Coordinate coordinate;
  Edge edge;
public:
  MatrixElement() = default;

  MatrixElement(const Coordinate& _coordinate, Edge _edge) : coordinate(_coordinate), edge(_edge) {}

  ~MatrixElement() = default;

  [[nodiscard]] Coordinate getCoordinate() const {
    return coordinate;
  }

  [[nodiscard]] Edge getEdgeValue() const {
    return edge;
  }

  void setCoordinate(const Coordinate& _coordinate) {
    coordinate = _coordinate;
  }

  void setEdgeValue(Edge _edge) {
    edge = _edge;
  }

  bool operator==(const MatrixElement &rhs) const {
    return coordinate == rhs.coordinate &&
           edge == rhs.edge;
  }

  bool operator!=(const MatrixElement &rhs) const {
    return !(rhs == *this);
  }

  bool operator<(const MatrixElement &rhs) const {
    if (coordinate < rhs.coordinate)
      return true;
    if (rhs.coordinate < coordinate)
      return false;
    return edge < rhs.edge;
  }

  bool operator>(const MatrixElement &rhs) const {
    return rhs < *this;
  }

  bool operator<=(const MatrixElement &rhs) const {
    return !(rhs < *this);
  }

  bool operator>=(const MatrixElement &rhs) const {
    return !(*this < rhs);
  }

};

#endif //SUBGRAPHISOMORPHISM_MATRIXELEMENT_H
