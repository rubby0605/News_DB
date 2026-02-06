#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: $0 no_ch_dir no_close_fd\n");
        exit(0);
    }

    int no_ch_dir = atoi(argv[1]);
    int no_close_fd = atoi(argv[2]);

    std::cout << "main pid : " << getpid() << std::endl;
    std::cout << "main parent pid : " << getppid() << std::endl;
    std::cout << "main pwd : " << get_current_dir_name() << std::endl;
    if (daemon(no_ch_dir, no_close_fd) != 0) {
        std::cerr << "stderr: daemon = -1" << std::endl;
        return 1;
    }
    std::cout << "stdout: daemon = 0" << std::endl;
    std::cerr << "stderr: daemon = 0" << std::endl;

    std::cout << "sub pid : " << getpid() << std::endl;
    std::cout << "sub parent pid : " << getppid() << std::endl;
    std::cout << "sub pwd : " << get_current_dir_name() << std::endl;
    while (1);
    return 0;
}

