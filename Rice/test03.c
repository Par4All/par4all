#include <stdlib.h>
#include <stdio.h>

/* Try casts*/

int main(int argc, char** argv){
  int a_d1, a_d2;
  int i, j;
  void * toto;

  a_d1=a_d2=5;
  toto = malloc(sizeof(double)*a_d1*a_d2);

  for(i=0;i<a_d1;i++){
    for(j=0;j<a_d2;j++){
      (*(double(*)[a_d1][a_d2])toto)[i][j] = (double) i+j*a_d2;
    }
  }

  free(toto);


  a_d1=a_d2=10;
  toto = malloc(sizeof(double)*a_d1*a_d2);

  for(i=0;i<a_d1;i++){
    for(j=0;j<a_d2;j++){
      (*(double(*)[a_d1][a_d2])toto)[i][j] = (double) i+j*a_d2;
    }
  }

  free(toto);


  return 0;
}
