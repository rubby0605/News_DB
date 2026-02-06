#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h> /* stat(), fstat() */
#include <string.h> /* memset() */
#include <fcntl.h> /* open() */
#include <time.h> /* localtime() */
#include <langinfo.h> /* nl_langinfo(), D_T_FMT */
#include <utime.h> /* utime() */

int cvt_time_to_date(time_t *time, char *date, unsigned int len)
{
	size_t		nchars;
	struct tm	*tm;
	if (time == NULL || date == NULL || len <=0)
		return(-4);

	tm = localtime(time);
	memset(date, 0, len);
	nchars = strftime(date, len, nl_langinfo(D_T_FMT), tm);
	if (nchars == 0)
		return(-5);
	else
		return(0);
}

int main(int argc, char * argv[])
{
	int ret;
	char *fname;
	struct stat finfo;
	char date[DATE_BUFSZ];
	struct utimbuf newtime;

	if( argc > 1)
		fname = argv[1];
	else
	{
		fprintf(stderr, "Usage : %s filename \n", argv[0]);
		return (-1);
	}
	ret = stat(fname, &finfo);
	if( ret != 0)
	{
		fprintf(stderr, "stat() failed, errno = %d\n", errno);
		return(-2);
	}
	fprintf(stdout, "time of last access is %ld \n", finfo.st_atime);
	ret = 






