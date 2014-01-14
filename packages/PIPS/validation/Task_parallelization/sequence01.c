#include <stdio.h>
#include <stdlib.h>
#include <time.h>



void main(int argc, char *argv[]){
  
  unsigned int i, N;
  double A[10], B[10], C[10];
  N = 10;
  for(i = 0 ; i < N ; i++){
    A[i] = 5; 
  }
  for(i = 0 ; i < N ; i++){
    B[i] = 3; 
  }
  for(i = 0 ; i < N ; i++)
    C[i]+= A[i] +  B[i]; 
}
     
      
