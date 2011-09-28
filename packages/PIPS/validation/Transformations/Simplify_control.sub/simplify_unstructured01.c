// test simplify_control when there is a declaration inside

#include <stdio.h>

int main(int argc, char* argv[])
{


  double _u_value1 = 0.0;
  double _u_value2 = 0.0;
  double _u_stock[3000];

  FILE * f = fopen("toto","r");
  /* READ FROM FILE */
  int _u_it = 1;
  while ( 1 ) {
    fscanf(f,"%lf", &_u_value1);
    fscanf(f,"%lf", &_u_value2);
    int _tmpxx5 = feof(f);
    if ((_tmpxx5!=0)) {
      break;
    }
    _u_stock[(_u_it-1)] = _u_value1 + _u_value2;
    _u_it = (_u_it+1);
  }
  fclose(f);

  for (int i=0; i<3000; i++)
    printf("%lf ", _u_stock[i]);
  return 0;
}

