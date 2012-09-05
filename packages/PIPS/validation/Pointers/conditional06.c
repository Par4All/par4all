/* Derived from list07, with code expansion and no list_free()
 *
 * Bug in the evaluation of the (dummy) conditional expression
 *
 * Bug in the update of the current points-to information because the
 * analysis of the conditional operator uses two separate context,
 * "int_t" and "in_f", to analyze the true and false sub-expressions.
 *
 * Note: the points-to analysis is limited because information gained
 * about c is not used for l, although c is initialized as l (see
 * PointerValues).
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
    // We know that c!=nil...
    // but this does not impact the result... as c->next may be NULL!
    c = (c==nil)? nil: c->next;
    n++;
  }
  return n;
}
