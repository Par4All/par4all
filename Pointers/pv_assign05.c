// assignment with subscripted and field expressions, imported from
// PointerValues

// Modified by Francois Irigoin to avoid nowhere/undefined everywhere

// Bug in the new version: fields on the right hand side are not
// dereferenced; here "s2.a = s1.a;" leads to s2.a -> s1.a, which is
// wrong

typedef struct {
  int *a;
  int *b[10];
  int (*c)[10];
} mystruct;

int main()
{
  mystruct s1, s2;
  mystruct tab_s[2];
  int i, j;
  s1.a = &i;
  s1.b[0] = &j;
  s2.a = s1.a;
  s2.b[0] = s1.b[0];
  tab_s[0].a = s1.a;
  tab_s[0].b[0] = s1.b[0];
  return(0);
}
