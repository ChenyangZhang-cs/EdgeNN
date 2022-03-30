#include <bits/stdc++.h>
#include <fstream>

using namespace std;

int readFile(char *fname, float *arr, int n) {
    ifstream ifile;
    ifile.open(fname, ios::in);
    if (!ifile) {
        cerr << "Open File Fail." << endl;
        return 1;
    }
    for (int i = 0; i < n; i++) {
        ifile >> arr[i];
    }
    ifile.close();
    return 0;
}

float sigmoid(float z) { return 1 / (1 + exp(-z)); }
