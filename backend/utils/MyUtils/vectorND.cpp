#include <cmath>
#include <ctime>
#include <iostream>
#include <utility>
#include <vector>
#include "vectorND.h"

using namespace std;

vector1D operator-(const vector1D& a, const vector1D& b) {
    vector1D c(a.size());
    for (int i = 0; i < (int)a.size(); i++) {
        c[i] = a[i] - b[i];
    }
    return c;
}

vector1D operator+(const vector1D& a, const vector1D& b) {
    vector1D c(a.size());
    for (int i = 0; i < (int)a.size(); i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

vector1D operator*(const vector1D& a, const double& b) {
    vector1D c(a.size());
    for (int i = 0; i < (int)a.size(); i++) {
        c[i] = a[i] * b;
    }
    return c;
}

vector1D operator*(const double& a, const vector1D& b) {
    return b * a;
}

vector1D operator/(const vector1D& a, const double& b) {
    vector1D c(a.size());
    for (int i = 0; i < (int)a.size(); i++) {
        c[i] = a[i] / b;
    }
    return c;
}

vector1D& operator+=(vector1D& a, const vector1D& b) {
    for (int i = 0; i < (int)a.size(); i++) {
        a[i] += b[i];
    }
    return a;
}

vector1D& operator-=(vector1D& a, const vector1D& b) {
    for (int i = 0; i < (int)a.size(); i++) {
        a[i] -= b[i];
    }
    return a;
}

vector1D& operator*=(vector1D& a, const double& b) {
    for (int i = 0; i < (int)a.size(); i++) {
        a[i] *= b;
    }
    return a;
}

vector1D& operator/=(vector1D& a, const double& b) {
    for (int i = 0; i < (int)a.size(); i++) {
        a[i] /= b;
    }
    return a;
}

vector1D get_zero1D(int size) {
    vector1D c(size);
    for (int i = 0; i < (int)size; i++) {
        c[i] = 0;
    }
    return c;
}

vector2D get_zero2D(int size1, int size2) {
    vector2D c;
    for (int i = 0; i < (int)size1; i++) {
        c.push_back(get_zero1D(size2));
    }
    return c;
}

vector3D get_zero3D(int size1, int size2, int size3) {
    vector3D c;
    for (int i = 0; i < (int)size1; i++) {
        c.push_back(get_zero2D(size2, size3));
    }
    return c;
}

double get_norm(const vector1D& v) {
    double norm = 0.0;
    for (int i = 0; i < (int)v.size(); i++) {
        norm += v[i] * v[i];
    }
    if (norm <= 0.0) {
        return 0.0;
    }
    return sqrt(norm);
}

vector1D get_unit_vector(const vector1D& v) {
    double norm = get_norm(v);
    if (norm < 1e-6) {
        return get_zero1D((int)v.size());
    } else {
        return v / norm;
    }
}

double vector_dot(const vector1D& a, const vector1D& b) {
    double dot = 0.0;
    for (int i = 0; i < (int)a.size(); i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

vector1D vector_dot(const vector2D& a, const vector1D& b) {
    vector1D c(a.size());
    for (int i = 0; i < (int)a.size(); i++) {
        c[i] = vector_dot(a[i], b);
    }
    return c;
}
