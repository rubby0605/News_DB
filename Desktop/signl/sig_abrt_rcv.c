#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

// 信号处理程序
void handle_sigabrt(int sig) {
    printf("Caught SIGABRT signal (%d). Abnormal termination detected.\n", sig);
    // 执行特定操作，例如记录日志或清理资源
    // ...

    // 终止程序
    exit(EXIT_FAILURE);
}

int main() {
    // 注册 SIGABRT 信号处理程序
    if (signal(SIGABRT, handle_sigabrt) == SIG_ERR) {
        perror("Failed to register SIGABRT handler");
        return 1;
    }

    // 模拟程序的主要工作
    printf("Running program... (will trigger SIGABRT)\n");
    sleep(2);

    // 触发 SIGABRT 信号
    abort();

    // 通常不会执行到这里，因为 abort() 会终止程序
    printf("This line will not be printed.\n");

    return 0;
}
