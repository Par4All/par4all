/* subset of list.c. Simplified version of list_len(): replace while
 * loop by a simple test
 *
 */

// a minimal list data structure

struct cons_t;
typedef struct cons_t * list;

// empty list
const list nil = ((list) 0);

// getter
list list_next(list);

// observer
int list_len(list);

struct cons_t {
  double value;
  list next;
};

// getter
// FI: double return...
list list_next(list nl)
{
  list n;
  if (nl==nil)
    n = nil;
  else
    n = nl->next;
  return n;
}

// observer
int list_len(list ll)
{
  int n = 0;
  list c = ll;
  //while (c!=nil)
  if (c!=nil)
  {
    // we know that c!=nil...
    c = list_next(c);
    n++;
  }
  return n;
}
