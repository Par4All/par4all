/* Make sure s is not redeclared in a cast */

struct s1
{
  int l;
};

void struct14()
{
  int i;
  void * p = (struct s1 *) &i;
  void * q = (struct s2 {int i; int j;} *) &i;
  void * r = (struct foo {int i; int j;} *) &i;
  struct s2 u2;
  struct foo u3;
}
