/* fmodl : These functions calculate the floating-point remainder of x/y. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void main()
{ 
   long double d1, d2, r; 
   printf("Enter two long doubles: "); 
   scanf("%lf %lf", &d1, &d2); 

   r = fmodl(d1, d2); 
   printf("fmod(%lf, %lf)=%lf\n", d1, d2, r); 
} 
