// to validate the regeneration of the call graph during simplify_control
// when there are calls inside variable declarations
#include <stdio.h>

double foo_fscanf(int iter, int fd, char* format)
{
  printf("%d",iter);
  printf("%d",fd);
  printf("%s",format);

  return iter;
}
int foo_eof(int fd)
{
  printf("%d", fd);
  return fd;
}

int foo_open(char* filename, char* flags)
{
  printf("%s",filename);
  printf("%s",flags);

  return filename[0];
}



int main()
{
  double _u_value1 = 0.0;
  double _u_value2 = 0.0;
  double _u_value3 = 0.0;
  char* _u_inputFile = "toto.c";
  int _u_fdIn = foo_open(_u_inputFile,"r");
  int _u_it = 1;
  while ( 1 ) {
    _u_value1 = foo_fscanf(1,_u_fdIn,"%lf");
    _u_value2 = foo_fscanf(1,_u_fdIn,"%lf");
    _u_value3 = foo_fscanf(1,_u_fdIn,"%lf");
    int _tmpxx5 = foo_eof(_u_fdIn);
    if ((_tmpxx5!=0)) {
      break;
    }
    _u_it = (_u_it+1);
  }

  return 0;
}
