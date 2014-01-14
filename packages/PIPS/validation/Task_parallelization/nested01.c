#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void main(int argc, char *argv[]){
  
  unsigned int i;
  double A[100], B[100], C[100], D[100];
  int N;
  double a, b;
  N = 100;
  for(i = 0 ; i < N ; i++){
    A[i] = 5;
    B[i] = 3;
  }
  for(i = 0 ; i < N ; i++)
    C[i] = 5;
  for(i = 0 ; i < N ; i++)
    D[i]= A[i] +  B[i] + C[i];
 
}


     
      
