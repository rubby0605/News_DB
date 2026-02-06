#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

// 信号处理程序
void handle_sigsegv(int sig) {
    printf("Caught SIGSEGV signal (%d). Segmentation fault detected.\n", sig);
    // 执行特定操作，例如记录日志或清理资源
    // ...

    // 终止程序
    exit(EXIT_FAILURE);
}

int main() {
    // 注册 SIGSEGV 信号处理程序
    if (signal(SIGSEGV, handle_sigsegv) == SIG_ERR) {
        perror("Failed to register SIGSEGV handler");
        return 1;
    }

    // 模拟程序的主要工作
    printf("Running program... (will cause segmentation fault)\n");

    // 触发分段错误
    int *p = NULL;
    *p = 42; // 尝试解引用空指针，触发分段错误

    // 通常不会执行到这里，因为程序会因为分段错误而被终止
    printf("This line will not be printed.\n");

    return 0;
}

