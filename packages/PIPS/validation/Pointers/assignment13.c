#include <stdio.h>
#include <stdlib.h>

struct foo {int * ip1; int * ip2;} ;

 void assignment(struct foo** t1, struct foo** t2) 
 {
   int *p;
   p = (**t2).ip2;
   (**t1).ip1 = p;
  } 
