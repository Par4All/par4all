/* Identify properly structure pointed to by expressions */

typedef struct s {
  int foo;
} a_t /* , *p_t*/;
typedef a_t * p_t;

typedef p_t f_t();

void point_to05()
{
  f_t *g;

  ((*g)()+2)->foo = 1;

  /* Floating point values are not allowed in pointer arithmetic */
  /* ((*g)()+2.)->foo = 1; */

  ((*g)()+(int)2.)->foo = 1;
  (2+(*g)())->foo = 1;
  ((*g)()-2)->foo = 1;

  /* This one is forbidden */
  /*(2-(*g)())->foo = 1; */
}
