/* Identify properly structure pointed to by expressions */

void point_to04()
{
  typedef struct s {
    int foo;
  } a_t /* , *p_t */;
  typedef a_t * p_t;

  a_t a;
  p_t p;

  (p+2)->foo = 1;

  /* Floating point values are not allowed in pointer arithmetic */
  /* (p+2.)->foo = 1; */

  (p+(int)2.)->foo = 1;
  (2+p)->foo = 1;
  (p-2)->foo = 1;

  /* This one is forbidden */
  /*(2-p)->foo = 1; */

  (p++)->foo = 1;
  (p--)->foo = 1;
  (++p)->foo = 1;
  (--p)->foo = 1;
}
