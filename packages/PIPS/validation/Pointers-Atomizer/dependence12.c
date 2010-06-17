/*****************************************************************************
 * XII - RECURSIVE STRUCTURE AS PARAM
 ****************************************************************************/
#include <stdlib.h>
struct my_struct {
  int a[10];
  struct my_struct *next;
};

void dependence12( struct my_struct *s ) {
  int *a;
  int i;

  // loop over a chained list
  while ( s != NULL ) {
    a = s->a; // a points_to "s[0][0][0]", but where does s points_to ?
    for ( i = 0; i < 10; i++ ) {
      a[i] = 0;
    }
    s = s->next;
  }
}
