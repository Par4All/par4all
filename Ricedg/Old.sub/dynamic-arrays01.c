#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <stdlib.h>

int N = 30000000;

float *array_a;
float *array_b;
float *array_c;
float *array_d;


int main(int argc, char **argv){
int i;

 if (argc > 1){
   N = atoi(argv[1]);
 }

 array_a = (float *) malloc(N*sizeof(float));
 assert(array_a!=NULL);
 array_b = (float *) malloc(N*sizeof(float));
 assert(array_b!=NULL);
 array_c = (float *) malloc(N*sizeof(float));
 assert(array_c!=NULL);
 array_d = (float *) malloc(N*sizeof(float));
 assert(array_d!=NULL);


 array_a [0]= 2.0f;
 array_b [0]= 2.0f;
 array_c [0]= 2.0f;

 for (i=1; i < N; i++){
   array_a[i] = 1.0f/((float)i);
   array_b[i] = 1.0f/((float)2*i);
   array_c[i] = 1.0f/((float)3*i);
 }


 for (i = 1; i < N; i++){
   array_a[i] = array_b[i] + array_c[i];
   array_d[i] = array_a[i-1] * 2;
 }

 for (i = 1; i < N; i++){
   printf("a[%d] = %f, d[%d] = %f\n", i, array_a[i], i, array_d[i]);
 }

 return 0;
}




