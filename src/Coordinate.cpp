//
// Created by z on 23-7-8.
//

#include <iostream>
#include "Coordinate.h"

Coordinate::Coordinate(int row, int col) : row(row), col(col) {}

Coordinate::~Coordinate() = default;

int Coordinate::getRow() const {
    return row;
}

int Coordinate::getCol() const {
    return col;
}

void Coordinate::setRow(int _row) {
    Coordinate::row = _row;
}

void Coordinate::setCol(int _col) {
    Coordinate::col = _col;
}

bool Coordinate::operator==(const Coordinate &rhs) const {
    return row == rhs.row &&
           col == rhs.col;
}

bool Coordinate::operator!=(const Coordinate &rhs) const {
    return !(rhs == *this);
}

bool Coordinate::operator<(const Coordinate &rhs) const {
    if (row < rhs.row)
        return true;
    if (rhs.row < row)
        return false;
    return col < rhs.col;
}

bool Coordinate::operator>(const Coordinate &rhs) const {
    if (row > rhs.row)
        return true;
    if (rhs.row > row)
        return false;
    return col > rhs.col;
}

bool Coordinate::operator<=(const Coordinate &rhs) const {
    return !(rhs < *this);
}

bool Coordinate::operator>=(const Coordinate &rhs) const {
    return !(*this < rhs);
}

Coordinate Coordinate::operator+(const Coordinate &rhs) const {
    return Coordinate(row + rhs.row, col + rhs.col);
}

Coordinate Coordinate::operator-(const Coordinate &rhs) const {
    return Coordinate(row - rhs.row, col - rhs.col);
}

Coordinate Coordinate::operator*(const Coordinate &rhs) const {
    return Coordinate(row * rhs.row, col * rhs.col);
}

Coordinate Coordinate::operator/(const Coordinate &rhs) const {
    return Coordinate(row / rhs.row, col / rhs.col);
}

Coordinate Coordinate::operator%(const Coordinate &rhs) const {
    return Coordinate(row % rhs.row, col % rhs.col);
}

Coordinate &Coordinate::operator+=(const Coordinate &rhs) {
    row += rhs.row;
    col += rhs.col;
    return *this;
}

Coordinate &Coordinate::operator-=(const Coordinate &rhs) {
    row -= rhs.row;
    col -= rhs.col;
    return *this;
}

Coordinate &Coordinate::operator*=(const Coordinate &rhs) {
    row *= rhs.row;
    col *= rhs.col;
    return *this;
}

Coordinate &Coordinate::operator/=(const Coordinate &rhs) {
    row /= rhs.row;
    col /= rhs.col;
    return *this;
}

Coordinate &Coordinate::operator%=(const Coordinate &rhs) {
    row %= rhs.row;
    col %= rhs.col;
    return *this;
}

Coordinate &Coordinate::operator++() {
    ++row;
    ++col;
    return *this;
}

const Coordinate Coordinate::operator++(int) {
    Coordinate tmp(*this);
    operator++();
    return tmp;
}

Coordinate &Coordinate::operator--() {
    --row;
    --col;
    return *this;
}

const Coordinate Coordinate::operator--(int) {
    Coordinate tmp(*this);
    operator--();
    return tmp;
}

std::ostream &operator<<(std::ostream &os, const Coordinate &coordinate) {
    os << "row: " << coordinate.getRow() << " col: " << coordinate.getCol();
    return os;
}


