#ifndef SUBGRAPHISOMORPHISM_GRAPHELEMENT_H
#define SUBGRAPHISOMORPHISM_GRAPHELEMENT_H

#include <string>

typedef int Node_t;
typedef int Edge_t;
typedef int Label_t;

class Node {
private:
  Node_t id;
  Label_t label;
public:
  Node() = default;

  Node(Node_t _id, Label_t _label) : id(_id), label(_label) {}

  ~Node() = default;

  [[nodiscard]] Node_t getId() const {
    return id;
  }

  void setId(int _id) {
    Node::id = _id;
  }

  [[nodiscard]] Label_t getLabel() const {
    return label;
  }

  void setLabel(const Label_t _label) {
    Node::label = _label;
  }

  bool operator==(const Node &rhs) const {
    return id == rhs.id;
  }

  bool operator!=(const Node &rhs) const {
    return !(rhs == *this);
  }

  bool operator<(const Node &rhs) const {
    if (id < rhs.id)
      return true;
    if (rhs.id < id)
      return false;
    return label < rhs.label;
  }

  bool operator>(const Node &rhs) const {
    if (id > rhs.id)
      return true;
    if (rhs.id > id)
      return false;
    return label > rhs.label;
  }

  bool operator<=(const Node &rhs) const {
    return !(rhs < *this);
  }

  bool operator>=(const Node &rhs) const {
    return !(*this < rhs);
  }
};

class Edge {
private:
  Edge_t id;
  Node_t src;
  Node_t dst;
  Label_t label;
public:
  Edge() = default;

  Edge(Edge_t _id, Node_t _src, Node_t _dst, Label_t _label) : id(_id), src(_src), dst(_dst), label(_label) {}

  ~Edge() = default;

  [[nodiscard]] Edge_t getId() const {
    return id;
  }

  void setId(Edge_t _id) {
    Edge::id = _id;
  }

  [[nodiscard]] Node_t getSrc() const {
    return src;
  }

  void setSrc(Node_t _src) {
    Edge::src = _src;
  }

  [[nodiscard]] Node_t getDst() const {
    return dst;
  }

  void setDst(Node_t _dst) {
    Edge::dst = _dst;
  }

  [[nodiscard]] Label_t getLabel() const {
    return label;
  }

  void setLabel(const Label_t _label) {
    Edge::label = _label;
  }

  bool operator==(const Edge &rhs) const {
    return id == rhs.id &&
           src == rhs.src &&
           dst == rhs.dst &&
           label == rhs.label;
  }

  bool operator!=(const Edge &rhs) const {
    return !(rhs == *this);
  }

  bool operator<(const Edge &rhs) const {
    if (id < rhs.id)
      return true;
    if (rhs.id < id)
      return false;
    if (src < rhs.src)
      return true;
    if (rhs.src < src)
      return false;
    if (dst < rhs.dst)
      return true;
    if (rhs.dst < dst)
      return false;
    return label < rhs.label;
  }

  bool operator>(const Edge &rhs) const {
    if (id > rhs.id)
      return true;
    if (rhs.id > id)
      return false;
    if (src > rhs.src)
      return true;
    if (rhs.src > src)
      return false;
    if (dst > rhs.dst)
      return true;
    if (rhs.dst > dst)
      return false;
    return label > rhs.label;
  }

  bool operator<=(const Edge &rhs) const {
    return !(rhs < *this);
  }

  bool operator>=(const Edge &rhs) const {
    return !(*this < rhs);
  }
};

#endif //SUBGRAPHISOMORPHISM_GRAPHELEMENT_H
