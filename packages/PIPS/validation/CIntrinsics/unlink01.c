#include <unistd.h>

int main()
{
  char *p1;
  p1 = "first_file_name";

  int i1 = unlink(p1);
  int i2= unlink("second_file_name");
  return (i1 + i2);
}
