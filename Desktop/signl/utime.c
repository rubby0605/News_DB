#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>  /*fctnl()*/
#include <fcntl.h> /*open()*/
#include <sys/stat.h>
#include <string.h> /*memset()*/
#include <stdlib.h> /*atoi()*/

#define FILENAME "./fcntl.data"
#define OFFSET2LOCK     (1024)
#define SIZE2LOCK       (1024)
#define MY_UPD_OFFSET   (1024)
#define MY_UPD_SIZE     (512)

int main(int argc, char * argv[])
{

    int     fd;
    struct  flock flock;
    char    buf[MY_UPD_SIZE];
    off_t   offset_ret, upd_offset, my_upd_offset;
    ssize_t bytes;
    size_t  count;
    int     ret;
    int     instance = 1;

    if (argc > 1)
        instance = atoi(argv[1]);
    if (instance < 1 || instance > 2)
    {
        fprintf(stderr, "Usage: %s [1 or 2]\n", argv[0]);
        return(-1);
    }

    fd = open(FILENAME, O_WRONLY);
    if (fd == 1)
    {
        fprintf(stderr, "Open file failed on %s, errno=%d\n", FILENAME, errno);
        return(-2);
    }
    flock.l_type = F_WRLCK; /* to obtain a write lock */
    flock.l_whence = SEEK_SET; /* offset relative to start of file */
    flock.l_start = OFFSET2LOCK; /* relative offset to lock */
    flock.l_len = SIZE2LOCK; /* number of bytes to lock */
    
    ret = fcntl(fd, F_SETLKW, &flock); /* Acquire the lock on the file segment */
    /* ret = fcntl(fd, F_SETLK, &flock); */
    if (ret == -1)
    {
        fprintf(stderr, "fcntl() failed to lock, errno = %d\n", errno);
        close(fd);
        return(-3);
    }
    for (instance = 1;instance <= 30;instance++)
    {
        fprintf(stdout, "Program instance %d got the file lock.\n", instance);
        if (instance == 1)
        {
            my_upd_offset = MY_UPD_OFFSET;
            my_upd_offset = my_upd_offset;
        }
        else
            my_upd_offset = my_upd_offset + 512;
        offset_ret = lseek(fd, my_upd_offset, SEEK_SET);
        if (offset_ret == -1)
        {
            fprintf(stderr, "lseek() failed, errno=%d", errno);
            close(fd); 
            return(-4);
        }
        memset((void *)buf, 'A'+(instance-1), MY_UPD_SIZE);
        bytes = write(fd, buf, MY_UPD_SIZE);
        if (bytes == 1)
        {
            fprintf(stderr, "write() failed, errno=%d", errno);
            close(fd);
            return(-5);
        }
        fprintf(stderr, "Program instance %d updates one block of file, but still hold on the lock...\n", instance);
        fprintf(stderr, "Program instance %d release the lock\n", instance);
        flock.l_type = F_UNLCK;
        ret = fcntl(fd, F_SETLK, &flock);
    
        if (ret == -1)
        {
fprintf(stderr, "fcntl() failed to lock, errno = %d\n", errno);
            close(fd);
            return(-6);
        }
    }
        close(fd);
    return(0);
}

