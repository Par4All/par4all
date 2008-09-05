/* Identify properly structure pointed to by expressions */

void point_to03()
{
  typedef struct s {
    int foo;
  } a_t /* , *p_t */;
  typedef a_t * p_t;
  a_t a;
  p_t p;

  (&a+2)->foo = 1;

  /* Floating point values are not allowed in pointer arithmetic */
  /* (&a+2.)->foo = 1; */

  (&a+(int)2.)->foo = 1;
  (2+&a)->foo = 1;
  (&a-2)->foo = 1;

  /* This one is forbidden */
  /*(2-&a)->foo = 1; */

  //((&a)++)->foo = 1;
  //((&a)--)->foo = 1;
  //(++(&a))->foo = 1;
  //(--(&a))->foo = 1;
}
