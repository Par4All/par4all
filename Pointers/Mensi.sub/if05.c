/* Same as if02.c, but without interferences with the heap model.
 *
 * To be studied by Francois: we still get imprceise information about
 * q at the end of main. Because of the exit, we should get an EXACT
 * arc.
 */

#include<stdio.h>
#include<stdlib.h>

void init(int* p)
{
  if(p == NULL)
    exit(1);
  else
    *p = 0;
}

int main()
{
  int init_p = 1 ;
  int *q = NULL;
  if(init_p)
    q = &init_p;
  init(q);
  return 0;
}
