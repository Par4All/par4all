#include <stdio.h>
#include <stdlib.h>

const int N=10;
const int Z=10;
int **Matrix;
char* filename;

int main (void) {
    int i,j;
    FILE *fp ;
    fp = fopen(filename,"r");
    if (fp == NULL)
    {
      printf(" 2 open the file");
      exit(1);
    }

    Matrix = (int **)malloc(N*sizeof(int *));
    for ( i = 0; i< N; i++)
    {
     	Matrix[i] = (int *)malloc(Z*sizeof(int));
       	for ( j = 0; j < Z; j++)
        {
         	fscanf(fp,"%d",&Matrix[i][j]);
        }

    }
    return 0;
}

