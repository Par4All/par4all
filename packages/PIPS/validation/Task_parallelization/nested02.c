#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/*The example of the chapter 7 of my thesis*/
void main(int argc, char *argv[]){
  
  unsigned int i, N;
  double A[10], B[10], C[10];
  double a,b;
  N = 10;
  for(i = 0 ; i < N ; i++){
    A[i] = 5; 
    B[i] = 3; 
  }
  for(i = 0 ; i < N ; i++){
    A[i]= A[i] * a; 
    C[i] = 0;       
  }
  for(i = 0 ; i < N ; i++)
    B[i]= B[i] * b; 
  for(i = 0 ; i < N ; i++)
    C[i]+= A[i] +  B[i]; 
  return;
}
     
      
