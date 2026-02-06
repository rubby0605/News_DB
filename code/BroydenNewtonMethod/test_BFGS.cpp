#include <iostream>
#include <cmath>
#include <vector>

// 定義目標函數 F1(x1, x2) = (x1 - 2)^2 + (x2 - 3)^2
double F1(double x1, double x2) {
    return std::pow(x1 - 2, 2) + std::pow(x2 - 3, 2);
}

// 定義目標函數 F3(x1, x2) = x1^2 + x2^2
double F3(double x1, double x2) {
    return std::pow(x1, 2) + std::pow(x2, 2);
}

// BFGS演算法實現
// 注意：這裡只是一個簡化的示例，真實情況下需要更完整的實現
// 傳入初始點 (x1, x2)，返回找到的最佳解 (best_x1, best_x2)
std::pair<double, double> BFGS(double x1, double x2) {
    const double tolerance = 1e-6;
    const int max_iterations = 1000;
    int iter = 0;
    double best_x1 = x1, best_x2 = x2;

    while (iter < max_iterations) {
        // Calculate gradient and Hessian (not shown here, assume they are available)
        double grad_x1 = 2 * (best_x1 - 2);
        double grad_x2 = 2 * (best_x2 - 3);

        // Update best_x1 and best_x2 using BFGS update rule
        // This part should implement the BFGS update rule

        iter++;

        // Check convergence criteria
        if (std::sqrt(grad_x1 * grad_x1 + grad_x2 * grad_x2) < tolerance) {
            break;
        }
    }

    return std::make_pair(best_x1, best_x2);
}

int main() {
    // 使用隨機初始值
    std::vector<std::pair<double, double>> initial_points = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0},
        {1.5, 2.5}, {2.5, 3.5}, {3.5, 4.5}, {4.5, 5.5}, {5.5, 6.5}
    };

    // 使用BFGS找到每個初始點的最佳解
    for (int i = 0; i < 10; ++i) {
        double x1 = initial_points[i].first;
        double x2 = initial_points[i].second;

        std::pair<double, double> best_solution = BFGS(x1, x2);

        // 輸出每個初始點找到的最佳解
        std::cout << "Initial point (" << x1 << ", " << x2 << "): ";
        std::cout << "Best solution found: (" << best_solution.first << ", " << best_solution.second << ")" << std::endl;
    }

    return 0;
}
