#ifndef SUBGRAPHISOMORPHISM_COORDINATE_H
#define SUBGRAPHISOMORPHISM_COORDINATE_H


class Coordinate {
private:
    int row, col;
public:
    Coordinate() = default;

    explicit Coordinate(int row = 0, int col = 0);

    ~Coordinate();

    [[nodiscard]] int getRow() const;

    [[nodiscard]] int getCol() const ;

    void setRow(int row);

    void setCol(int col);

    bool operator==(const Coordinate &rhs) const;

    bool operator!=(const Coordinate &rhs) const;

    bool operator<(const Coordinate &rhs) const;

    bool operator>(const Coordinate &rhs) const;

    bool operator<=(const Coordinate &rhs) const;

    bool operator>=(const Coordinate &rhs) const;

    Coordinate operator+(const Coordinate &rhs) const;

    Coordinate operator-(const Coordinate &rhs) const;

    Coordinate operator*(const Coordinate &rhs) const;

    Coordinate operator/(const Coordinate &rhs) const;

    Coordinate operator%(const Coordinate &rhs) const;

    Coordinate &operator+=(const Coordinate &rhs);

    Coordinate &operator-=(const Coordinate &rhs);

    Coordinate &operator*=(const Coordinate &rhs);

    Coordinate &operator/=(const Coordinate &rhs);

    Coordinate &operator%=(const Coordinate &rhs);

    Coordinate &operator++();

    const Coordinate operator++(int);

    Coordinate &operator--();

    const Coordinate operator--(int);

    Coordinate operator-() const;

    Coordinate operator+() const;

    Coordinate operator!() const;

    Coordinate operator~() const;

    Coordinate operator&(const Coordinate &rhs) const;


};


#endif //SUBGRAPHISOMORPHISM_COORDINATE_H
