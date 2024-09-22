#include "ufs.h"
#include <iostream>
#include <map>
#include <vector>

using namespace std;

UFS::UFS(int n1) : n(n1) {
    parent.resize(n1);
    size.resize(n1, 1);
    for (int i = 0; i < n1; i++) {
        parent[i] = i;
    }
}

int UFS::find(int i) {
    return parent[i] == i ? i : parent[i] = find(parent[i]);
}

bool UFS::connected(int i, int j) {
    return find(i) == find(j);
}

void UFS::union_set(int i, int j) {
    // parent[find(i)] = find(j);
    int pi = find(i);
    int pj = find(j);
    if (pi != pj) {
        parent[pi] = pj;
        size[pj] += size[pi];
    }
}

int UFS::get_size(int i) {
    // get the size of the set that i belongs to
    return size[find(i)];
}


vector<vector<int>> UFS::get_group_indicator() {
    for (int i = 0; i < n; i++) {
        find(i);
    }

    map<int, int> group_map;
    int group_n = 0;
    for (int i = 0; i < n; i++) {
        if (group_map.count(parent[i]) == 0) {
            group_map[parent[i]] = group_n++;
        }
    }

    vector<vector<int>> group_indicator;
    for (int i = 0; i < n; i++) {
        group_indicator.push_back(vector<int>(group_n));
        group_indicator[i][group_map[parent[i]]] = 1;
    }

    return group_indicator;
}

vector<vector<int>> UFS::get_group() {
    for (int i = 0; i < n; i++) {
        find(i);
    }

    map<int, int> group_map;
    int group_n = 0;
    for (int i = 0; i < n; i++) {
        if (group_map.count(parent[i]) == 0) {
            group_map[parent[i]] = group_n++;
        }
    }

    vector<vector<int>> group;
    for (int i = 0; i < group_n; i++) {
        group.push_back(vector<int>());
    }

    for (int i = 0; i < n; i++) {
        group[group_map[parent[i]]].push_back(i);
    }

    return group;
}
