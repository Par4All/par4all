#include <stdio.h>



int main(int argc, char **argv)
{
   int nx;
   nx = 1000;
   int ny;
   ny = 1000;
   int ey[nx][ny];

   int t1, t3;

   if (nx>=2&&ny==1) {
     for(t1 = 0; t1 < nx; t1 += 1) {
       ey[0][0] = 0;
       // this loop should not be parallelized, because it is dead code
       for(t3 = 0; t3 < ny; t3 += 1)
         ey[t1][t3] = 0;
     }
   }

   // Same loop nest but not surrounded by if, and thus not dead code:
   // the innermost loop is parallel
   for(t1 = 0; t1 < nx; t1 += 1) {
     ey[0][0] = 0;
     for(t3 = 0; t3 < ny; t3 += 1)
       ey[t1][t3] = 0;
   }


   if (nx>=2&&ny==1) {
       // these loops should not be parallelized, because they are dead code
      for(t1 = 0; t1 < nx; t1 += 1) {
       // ey[0][0] = 0;
       for(t3 = 0; t3 < ny; t3 += 1)
         ey[t1][t3] = 0;
     }
   }

   printf("%d\n",ey[0][0]);

   return 0;
}
