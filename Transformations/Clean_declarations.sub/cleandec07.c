// The declaration of array c is useless, but not the expression used
// to size it

int c_clean_declarations07(int x)
{
  int a[x];
  int b = sizeof(struct s07 {int r; int i;});
  struct s07 c;
  return c.r;
}
