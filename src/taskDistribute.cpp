#include <iostream>
using namespace std;

int main() {
    double volumn, bandwidth;
    double t_cpu, t_gpu;
    double p_cpu = 0;
    cout << "input the data volumn and transfer bandwidth\n";
    cin >> volumn >> bandwidth;
    cout << "input the execution time of the CPU and the GPU\n";
    cin >> t_cpu >> t_gpu;

    if (volumn / bandwidth >= t_gpu)
        p_cpu = 0;
    else
        p_cpu = t_gpu / (t_cpu + t_gpu);

    cout << "Optimal porprotion of CPU is " << p_cpu << endl;
}