#define SIZE 10
double get(double f[SIZE],int i)
{
    return f[i];
}

void foo(double A[SIZE], double B[SIZE][SIZE])
{
    int i,j;
l0:    for(i=0;i<SIZE;i++)
       {
l1:        for(j=0;j<SIZE;j++)
           {
               A[i] = B[j][i] + get(A,i);
           }
       }
}



