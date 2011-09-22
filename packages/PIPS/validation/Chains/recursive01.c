/* Check that parallel loops using recursive data types are found.

   Check that type_depth() is OK with recursive data types.

   To be improved with a real list and a while loop inside?
 */

  typedef struct al {
    float x[00];
    struct al * next;
  } al_t;

void foo(al_t e)
{
  ;
}

void for03(int n)
{
  int j;
  al_t e1;
  al_t e2;
  float t, delta_t, t_max;

  e1.next = &e2;

  for(j=0;j<100;j++) {
    e1.x[j] = 0.;
    e1.next->x[j] = 1.;
  }
  //foo(e1);
}
