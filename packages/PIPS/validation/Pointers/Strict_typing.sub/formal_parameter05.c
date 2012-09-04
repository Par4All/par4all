/* Check recursive stub generation and struct declarations
 */

typedef struct foo {
  int one[10];
  struct foo * two;
} my_struct;

int formal_parameter05(my_struct *p)
{
  my_struct s1 = *p;
  my_struct s2;
  my_struct s3;
  s2 = s1;
  s3 = *p;
  return 0;
}
