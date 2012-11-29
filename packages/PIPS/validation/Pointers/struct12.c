/* Check struct assignments
 *
 * s4 and assignment of s4 to s1.three added to avoid an error
 * detection in assignment "s2=*p;".
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
  my_struct s2, s3, s4;
  my_struct *p = &s1;
  s1.two = &s3;
  s1.three = &s4;
  s2 = *p;

  return 0;
}
