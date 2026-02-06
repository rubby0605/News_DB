#include <iostream>
#include <vector>
#include <algorithm>

struct State {
    double voltage;  // 电压
    double frequency; // 频率
    double power;    // 功耗
    double executionTime; // 执行时间
};

double totalPower(const std::vector<State>& states) {
    double totalPower = 0;
    for (const auto& state : states) {
        totalPower += state.power;
    }
    return totalPower;
}

std::vector<State> greedyDVFS(const std::vector<State>& states, double maxTime) {
    std::vector<State> selectedStates;
    double totalTime = 0;

    // 按照功耗升序排序
    std::vector<State> sortedStates = states;
    std::sort(sortedStates.begin(), sortedStates.end(), [](const State& a, const State& b) {
        return a.power < b.power;
    });

    // 贪心选择状态
    for (const auto& state : sortedStates) {
        if (totalTime + state.executionTime <= maxTime) {
            selectedStates.push_back(state);
            totalTime += state.executionTime;
        }
    }

    return selectedStates;
}

int main() {
    // 假设有以下状态
    std::vector<State> states = {
        {1.0, 1.0, 1.0, 1.0},  // 电压1V，频率1GHz，功耗1W，执行时间1s
        {0.9, 0.9, 0.81, 1.1}, // 电压0.9V，频率0.9GHz，功耗0.81W，执行时间1.1s
        {0.8, 0.8, 0.64, 1.3}, // 电压0.8V，频率0.8GHz，功耗0.64W，执行时间1.3s
        {1.1, 1.1, 1.21, 0.9}, // 电压1.1V，频率1.1GHz，功耗1.21W，执行时间0.9s
    };

    double maxTime = 3.0; // 最大执行时间为3秒

    auto selectedStates = greedyDVFS(states, maxTime);

    std::cout << "Selected States:\n";
    for (const auto& state : selectedStates) {
        std::cout << "Voltage: " << state.voltage 
                  << "V, Frequency: " << state.frequency 
                  << "GHz, Power: " << state.power 
                  << "W, Execution Time: " << state.executionTime << "s\n";
    }

    double totalPowerUsed = totalPower(selectedStates);
    std::cout << "Total Power Used: " << totalPowerUsed << "W\n";

    return 0;
}

