/* Check struct assignments
 */

struct foo {
  int one[10];
  struct foo * two;
  struct foo * three;
};

typedef struct foo my_struct;

int struct12()
{
  struct foo s1;
  my_struct s2;
  my_struct s3;
  my_struct *p = &s1;
  s1.two = &s3;
  s2 = *p;

  return 0;
}
