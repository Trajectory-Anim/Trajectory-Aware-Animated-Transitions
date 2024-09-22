#pragma once

#include <vector>

using namespace std;

class UFS {
   private:
    vector<int> parent;
    vector<int> size;
    int n;

   public:
    UFS(int n1);
    int find(int i);
    bool connected(int i, int j);
    void union_set(int i, int j);
    int get_size(int i);
    vector<vector<int>> get_group_indicator();
    vector<vector<int>> get_group();
};
