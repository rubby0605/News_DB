#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#define MATRIX_SIZE 1000000
#define NUM_THREADS 10

int *matrix;
pthread_t threads[NUM_THREADS];
void read_matrix(void *arg);
pthread_mutex_t mutex;
void record_log(void *arg);

typedef struct {
    int start;
    int end;
} ThreadData;

void record_log(void *arg){
	//start writing log
	fprintf(stderr, "Start to write logs\n");
}

void read_matrix(void *arg){
	ThreadData *data = (ThreadData *) arg;
	long sum = 0;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
		pthread_mutex_lock(&mutex);
		sum += matrix[i];
		pthread_mutex_unlock(&mutex);
	}
    printf("Thread %lu: Sum from index %d to %d is %ld\n", pthread_self(), data->start, data->end, sum);
}

// 信号处理程序 signal abort
void handle_sigabrt(int sig) {
    printf("Caught SIGABRT signal (%d). Abnormal termination detected.\n", sig);
    // record log
    record_log("This is log");

    // 终止程序
    exit(EXIT_FAILURE);
}

// 信号处理程序 signal abort
void handle_sigkill(int sig) {
    printf("Caught SIGKILL signal (%d). Abnormal termination detected.\n", sig);
    // record log
    record_log("This is log");

    // 终止程序
    exit(EXIT_FAILURE);
}

// 信号处理程序interupt sginal
void handle_sigint(int sig) {
    printf("Caught SIGSEGV signal (%d). Abnormal termination detected.\n", sig);
    // record log
    record_log("This is log");

    // 终止程序
    exit(EXIT_FAILURE);
}
// 信号处理程序segmentation faullt
void handle_sigsegv(int sig) {
    printf("Caught SIGSEGV signal (%d). Abnormal termination detected.\n", sig);
    // 执行特定操作，例如记录日志或清理资源
    // ...

    // 终止程序
    exit(EXIT_FAILURE);
}

int main() {
	int i;
    srand(time(NULL));
    if (signal(SIGABRT, handle_sigabrt) == SIG_ERR) {// should put in the first line!!
    perror("Failed to register SIGABRT handler");
    	return 0;
    }
    if (signal(SIGSEGV, handle_sigsegv) == SIG_ERR) {
    	perror("Failed to register SIGSEGV handler");
    	return 0;
    }
    if (signal(SIGKILL, handle_sigkill) == SIG_ERR) {
    	perror("Failed to register SIGKILL handler");
    	return 0;
    }
    if (signal(SIGINT, handle_sigsegv) == SIG_ERR) {
    	perror("Failed to register SIGINT handler");
    	return 0;
    }

    // 分配和初始化矩阵
    matrix = (int*)malloc(MATRIX_SIZE * sizeof(int));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    matrix[i] = rand() % 100;
	sleep(10);

    // 创建线程
    ThreadData thread_data[NUM_THREADS];
    int elements_per_thread = MATRIX_SIZE / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
		//f(SIGCONT
        thread_data[i].start = i * elements_per_thread;
        thread_data[i].end = (i == NUM_THREADS - 1) ? MATRIX_SIZE : (i + 1) * elements_per_thread;
        pthread_create(&threads[i], NULL, read_matrix, &thread_data[i]);
    }

    // 等待线程完成
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    // 释放矩阵内存
    free(matrix);

    return 0;
}
