#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>

int main(void)
{
pid_t pid;
int i=0, j=12;
for (i=0;i<=12;i++)
{
	pid = fork();
	std::cout<<"pid is "<<pid;
}
if((i=j)>0) std::cout<<i;

}


