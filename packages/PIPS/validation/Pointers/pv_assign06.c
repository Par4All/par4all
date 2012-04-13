// assignments which kill aliased paths 
typedef struct {int *a; int *b[10]; int (*c)[10];} mystruct;
int main()
{
  mystruct s1, *s2;
  int i = 1;
  int j = 2;
  s2 = &s1;
  s2->a = &i;
  s1.a = &j;
  s2->b[0] = &i;
  return(0);
}
