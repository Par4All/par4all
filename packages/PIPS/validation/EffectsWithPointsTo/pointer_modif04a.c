/* Struct variable are passed by value...
 *
 * Function init_s does not work because s is passed by copy. The
 * initializations are lost on return.
 *
 * Several bugs shown here linked to the subscript and field operations
 *
 * Here we have both: a field intrinsic and a subscript construct
 */

typedef struct {int max; float *tab;} s_t;

void compute_s(s_t s, int max)
{
  int i;

  for (i=0; i<max; i++)
    s.tab[i] = i*2.0;
  
  return;
}
