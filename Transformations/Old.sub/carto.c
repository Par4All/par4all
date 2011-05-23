#include <math.h>
float dist(int x0, int y0, int x1, int y1)
{
   int X = x0-x1;
   int Y = y0-y1;
   return sqrt(X*X+Y*Y);
}
void carto(float towns[100][100], float result[100][100])
{
   int i;
   int j;
   int k;
   int l;
l0:   for(i = 0; i < 100; i ++)
l1:   for(j = 0; j < 100; j ++) {
         result[i][j] = towns[i][j];
l2:      for(k = 0; k < 100; k++)
l3:       for(l = 0; l < 100; l++)
               if (i!=k&&j!=l) {
                  result[i][j] += towns[k][l]/dist(i,j,k,l);
               }
      }
}
int main(int argc, char *argv[])
{
   float T[100][100];
   float R[100][100];
   float f = 0;
   int i;
   int j;
   for(i = 0; i <= 99; i += 1)
      for(j = 0; j <= 99; j += 1)
         T[i][j] = i*j;
   

   carto(T, R);
   for(i = 0; i <= 99; i += 1)
      for(j = 0; j <= 99; j += 1)
         f += R[i][j];
   printf("%f\n", f);
   return 0;
}
