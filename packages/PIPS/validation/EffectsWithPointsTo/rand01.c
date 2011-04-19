#include <stdlib.h>             /* srand rand */
 
int main ()
{
   int entier;
   int graine;
   double virgule;
   graine=50;
   srandom (graine);         /* generator initialization */
   virgule = (double) rand () / (RAND_MAX+1);
   entier = 1 + rand ();
}
