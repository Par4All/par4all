/* Beware of comma expressions used as subscripts */

int main() {
  int b[10];
  int i=0,j=1;

  // Unlike Fortran, this is only a one dimensionnal access
  // Pips used to parse it like b[i][j]
  b[i,j] = 0;

}


