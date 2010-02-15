/* This C code is not legal according to gcc's warning because the
   scope of u_u declaration is limited, but no error is reported when
   u_u is used out of this scope.

   Furthermore, the PIPS parser detect a conflict between the member i
   and the parameter i!
   */

struct s
{
  int l;
};

void struct11(union u_u
{
  struct s d1;
  struct s d2;
  int i;
} u, int /* i */ j)
{
  int j = sizeof(union u_u);
  union u_u u2;
}
