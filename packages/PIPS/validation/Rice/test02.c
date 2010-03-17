#include <stdlib.h>
#include <stdio.h>

// it doesn' t work : the dependent type definition of toto is frozen

int main(int argc, char** argv){
  int a_d1,a_d2;
  int i,j;
  double *toto;

  a_d1=a_d2=5;
  toto = malloc(sizeof(*toto)*a_d1*a_d2);

  for(i=0;i<a_d1;i++){
    for(j=0;j<a_d2;j++){
      toto[i+ j*a_d1]=i+j*a_d2;
    }
  }

  free(toto);


  a_d1=a_d2=10;
  toto = malloc(sizeof(*toto)*a_d1*a_d2);

  for(i=0;i<a_d1;i++){
    for(j=0;j<a_d2;j++){
      toto[i+ j*a_d1]=i+j*a_d2;
    }
  }

  free(toto);


  return 0;
}
