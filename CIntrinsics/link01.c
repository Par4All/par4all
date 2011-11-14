#include <unistd.h>

int main()
{
  char *p1;
  p1 = "first_file_name";

  int i1 = link(p1, "second_file_name");
  return (i1);
}
