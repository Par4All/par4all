/* Same as global03.c, but different tpips script 
 *
 * Check points-to analysis for computation unit, initial points-to
 * and program points-to.
 */

typedef struct two_fields{int one; int two[10];} tf_t;

int i[10];
tf_t s;
int *pi = &i[0];
tf_t *q = &s;

int global03a()
{
  int * p = i;
  *pi = 1;
  pi++;
  q->one = 1;
  q->two[4] = 2;
  return *p;
}
