// assignment with subscripted and field expressions 
typedef struct {int *a; int *b[10]; int (*c)[10];} mystruct;
int main()
{
  mystruct s1, s2;
  mystruct tab_s[2];
  s2.a = s1.a;
  s2.b[0] = s1.b[0];
  tab_s[0].a = s1.a;
  tab_s[0].b[0] = s1.b[0];
  return(0);
}
