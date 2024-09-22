#pragma once

#include <iostream>
#include <utility>
#include <vector>

using namespace std;

typedef vector<double> vector1D;
typedef vector<vector1D> vector2D;
typedef vector<vector2D> vector3D;
typedef vector<vector3D> vector4D;

vector1D operator-(const vector1D& a, const vector1D& b);
vector1D operator+(const vector1D& a, const vector1D& b);
vector1D operator*(const vector1D& a, const double& b);
vector1D operator*(const double& a, const vector1D& b);
vector1D operator/(const vector1D& a, const double& b);

vector1D& operator+=(vector1D& a, const vector1D& b);
vector1D& operator-=(vector1D& a, const vector1D& b);
vector1D& operator*=(vector1D& a, const double& b);
vector1D& operator/=(vector1D& a, const double& b);

vector1D get_zero1D(int size);
vector2D get_zero2D(int size1, int size2);
vector3D get_zero3D(int size1, int size2, int size3);

double get_norm(const vector1D& v);
vector1D get_unit_vector(const vector1D& v);

double vector_dot(const vector1D& a, const vector1D& b);
vector1D vector_dot(const vector2D& a, const vector1D& b);
