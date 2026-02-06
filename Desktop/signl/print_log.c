#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>

// 定义日志文件名
#define LOG_FILE "output.log"

// 自定义 printf 函数，既打印到终端又写入日志文件
void log_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);

    // 打开日志文件（以追加模式）
    FILE *log_file = fopen(LOG_FILE, "a");
    if (log_file == NULL) {
        perror("Failed to open log file");
        exit(EXIT_FAILURE);
    }

    // 获取当前时间
    time_t now;
    time(&now);
    struct tm *local = localtime(&now);
    fprintf(log_file, "[%02d-%02d-%04d %02d:%02d:%02d] ",
            local->tm_mday, local->tm_mon + 1, local->tm_year + 1900,
            local->tm_hour, local->tm_min, local->tm_sec);

    // 将输出写入日志文件
    vfprintf(log_file, format, args);

    // 将输出打印到终端
    vprintf(format, args);

    // 关闭日志文件
    fclose(log_file);

    va_end(args);
}

int main() {
    // 示例输出
    log_printf("This is a test message.\n");
    log_printf("The value of variable x is %d.\n", 42);
    log_printf("Another message with a float: %f.\n", 3.14159);

    return 0;
}

