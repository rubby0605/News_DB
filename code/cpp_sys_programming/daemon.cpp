#include <fcntl.h>
#include <string.h>
#include <algorithm>

//#include "dictionary_exclude.h"
#pragma GCC diagnostic ignored "-Wwrite-strings"

#define CHECK_THROW(condition, code) if(condition) throw code
void *SocketHandler(void *);
int OpenSocket();
int main(int argv, char **argc)
{
    int host_port = 1104;
    char buf[20];
    int k;
    struct sockaddr_in my_addr;
    int hsock;
    int *p_int;
    int err;
    socklen_t addr_size = 0;
    int *csock;
    sockaddr_in sadr;
    pthread_t thread_id = 0;

    try {
    pid_t pid = fork();
    CHECK_THROW(pid < 0, -5);
    if (pid == 0) {
        mode_t umask(mode_t mask);
        pid_t childid = setsid();

        hsock = OpenSocket();   // Function call for hsock
        CHECK_THROW(listen(hsock, 10) == -1, -4);
        my_addr.sin_family = AF_INET;
        my_addr.sin_port = htons(host_port);

        memset(&(my_addr.sin_zero), 0, 8);
        my_addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(hsock, (sockaddr *) & my_addr, sizeof(my_addr)) == -1) {
        fprintf(stderr, "Error binding to socket, make sure nothing else is listening on this port %dn", errno);
        exit(EXIT_SUCCESS);
        }
        if (listen(hsock, 10) == -1) {
        fprintf(stderr, "Error listening %dn", errno);
        exit(EXIT_SUCCESS);
        }
        //Now lets do the server stuff

        addr_size = sizeof(sockaddr_in);

        while (true) {
        printf("waiting for a connectionnn");
        csock = (int *) malloc(sizeof(int));
        if ((*csock = accept(hsock, (sockaddr *) & sadr, &addr_size)) != -1) {
            printf("---------------------nReceived connection from %sn", inet_ntoa(sadr.sin_addr));
            pthread_create(&thread_id, 0, &SocketHandler, (void *) csock);
            pthread_detach(thread_id);
        } else {
            fprintf(stderr, "Error accepting %dn", errno);
        }
        }           //while end
    }           //if (pid) end
    }               //try

    catch(int ierror) {
    switch (ierror) {
    case -4:
        fprintf(stderr, "Error listening %dn", errno);
        break;
    case -7:
        fprintf(stderr, "Error accepting %dn", errno);
        break;
    }
    }

}

int OpenSocket()
{
    // Create your socket and return the socket handle from this function 
    int hsock;
    int *p_int;
    hsock = socket(AF_INET, SOCK_STREAM, 0);
    if (hsock == -1) {
    printf("Error initializing socket %dn", errno);
    exit(EXIT_SUCCESS);
    }

    p_int = (int *) malloc(sizeof(int));
    *p_int = 1;

    if ((setsockopt(hsock, SOL_SOCKET, SO_REUSEADDR, (char *) p_int, sizeof(int)) == -1) || (setsockopt(hsock, SOL_SOCKET, SO_KEEPALIVE, (char *) p_int, sizeof(int)) == -1)) {
    printf("Error setting options %dn", errno);
    free(p_int);
    exit(EXIT_SUCCESS);
    }
    free(p_int);
    return hsock;
}


void *SocketHandler(void *lp)
{
for(int i;i<=100;i++)
{
std::cout<<i;
}
    //some procesing
}

