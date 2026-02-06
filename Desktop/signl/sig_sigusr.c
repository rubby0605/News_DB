#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>  /*fctnl()*/
#include <fcntl.h> /*open()*/
#include <sys/stat.h>
#include <string.h> /*memset()*/
#include <stdlib.h> /*exit()*/
#include <signal.h>

int signal_handler(int sig)
{
	fprintf(stdout, " In signal_handler, this process received a signal %d\n", sig);
	fprintf(stdout, "shutting down..");
	exit(0);
}
int main(int argc, char argv[]){
	pid_t pid;
	int stat;
	int ret;
	pid = fork();
	if( pid == -1){
		fprintf(stderr, " fork failed, errno = %d", errno);
	return(-1);
	}
	else if (pid == 0){
		/* This is the child process */
		struct sigaction newact, oldact;
		fprintf(stdout);
		
		/*Specify an action for a signal*/
		sigfillset(&newact.sa_mask);
		newact.sa_flags = 0;

		/* Specify an own function (like install or something) */
		newact.handler = (void (*)(int))signal_handler;

	}

} 
