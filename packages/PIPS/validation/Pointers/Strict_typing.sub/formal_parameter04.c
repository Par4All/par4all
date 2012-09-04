/* Check recursive stub generation */

typedef struct foo {
  int one[10];
  struct foo * two;
} my_struct;

int formal_parameter04(my_struct *p)
{
  p->one[0]=0;
  p->two->one[0]=0;
  p->two->two->one[0]=0;
  p->two->two->two->one[0]=0;
  p->two->two->two->two=0;
  return 0;
}
