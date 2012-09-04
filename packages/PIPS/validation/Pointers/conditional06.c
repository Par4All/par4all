/* Derived from list07, with code expansion and no list_free()
 *
 * Bug in the evaluation of the (dummy) conditional expression
*/

#include <stdlib.h>

struct cons_t;
typedef struct cons_t * list;

// empty list
const list nil = ((list) 0);

struct cons_t {
  double value;
  list next;
};

int list_len(list l)
{
  int n = 0;
  list c = l;
  //while (c!=nil) {
  if (c!=nil) {
    // we know that c!=nil...
    c = (c==nil)? nil: c->next;
    n++;
  }
  return n;
}
