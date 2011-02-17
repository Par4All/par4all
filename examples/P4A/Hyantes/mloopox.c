/*
 * file for mloop.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct {
   float latitude;
   float longitude;
   float stock;
} town;

int main();
int main()
{
   size_t i, j, k;
   float xmin, ymin, step, range;
   int rangex = 300;
   int rangey = 300;
   int nb = 3000;
   town pt[300][300], t[3000];

   xmin = ymin = step = range = 2.0;
#pragma omp parallel for private(k)
   for(k = 0; k <= nb-1; k += 1) {
      t[k].latitude = k*step*180/3.14159265358979323846;
      t[k].longitude = k*step*180/3.14159265358979323846;
      t[k].stock = 500.;
   }

#pragma omp parallel for private(k, j, i)
   for(i = 0; i <= rangex-1; i += 1)
#pragma omp parallel for private(k, j)
      for(j = 0; j <= rangey-1; j += 1) {
         pt[i][j].latitude = (xmin+step*i)*180/3.14159265358979323846;
         pt[i][j].longitude = (ymin+step*j)*180/3.14159265358979323846;
         pt[i][j].stock = 0.;
#pragma omp parallel for private(k)
         for(k = 0; k <= nb-1; k += 1) {
            float tmp = 6368.*acos(cos(xmin+step*i)*cos(t[k].latitude)*cos(ymin+step*j-t[k].longitude)+sin(xmin+step*i)*sin(t[k].latitude));
            if (tmp<range)
               pt[i][j].stock += t[k].stock/(1+tmp);
         }
      }

   for(i = 0; i <= rangex-1; i += 1)
      for(j = 0; j <= rangey-1; j += 1)
         printf("%f %f %f\n", pt[i][j].latitude, pt[i][j].longitude, pt[i][j].stock);

   return 0;
}
