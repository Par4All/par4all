
void lud_99_kernel(int size, int i, float a[size][size])
{
   int j, k;
   for (j=i; j<size; j++){
     float sum=a[i][j];
     for (k=0; k<i; k++) {
       sum -= a[i][k]*a[k][j];
     }
     a[i][j]=sum;
   }
}



void lud_99_kernel_flatten(int size, int i, float sum, float a[size][size])
{
   int j, k;
   for (j=i; j<size; j++){
     sum=a[i][j];
     for (k=0; k<i; k++) {
       sum -= a[i][k]*a[k][j];
     }
     a[i][j]=sum;
   }
}


