/* Test case designed by Fabien Coelho to discuss fix point issues for
 * the points-to analysis
 *
 * The function make a reverse copy of the input list
 *
 * This version is buggy. Memory leaks must be detected.
 */

// For the definition of NULL
#include <stdio.h>
// For the definition of malloc()
#include <stdlib.h>

// struct rec_st;

//typedef struct rec_st * rec;

struct rec_st {
  int i;
  struct rec_st * n;
};

typedef struct rec_st * rec;

rec reverse_copy(rec p)
{
  rec r = NULL;
  rec q = p;
  while(q!=NULL) {
    rec t = (rec) malloc(sizeof(struct rec_st));
    t->i = q->i;
    t->n = r;
    q = q->n;
  }
  return r;
}
