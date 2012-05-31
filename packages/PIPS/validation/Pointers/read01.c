#include <sys/types.h>
#include <unistd.h>

int main()
{
  char buf[20];
  size_t nbytes;
  ssize_t bytes_read;
  int fd;

  nbytes = sizeof(buf);
  bytes_read = read(fd, buf, nbytes);
  return 0;
}
