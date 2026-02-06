#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h> ///so library usage

int g1;//same name condition : use the so global one instead of this one

int main(void)
{
	char *so_name = "/Users/rubylintu/python_embeded_c/f1.so";

	void *handle = dlopen (so_name, RTLD_NOW | RTLD_GLOBAL );
        //RTLD_LAZY can be use, too
	if ( !handle )
	{
		fprintf( stderr, "[Error] dlopen '%s' fail --> %s \n",so_name, dlerror() );
		exit(1);
	}
	
	double (*p_f1) (double) = (double (*) (double)) dlsym (handle, "f1" ) ;
	if ( !p_f1 )
	{
		fprintf(stderr, "[Error] lsym 'f1' fail --> %s\n", dlerror() );
		exit(1);
	}

	double x = 10;
	printf ("f1(x) %.15le\n",p_f1(x));
	return 0;
}
