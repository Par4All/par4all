// assignment of aggregate structure 

// FI: modified so as not to assign undefined pointers

typedef struct {int *a; int *b[10]; int (*c)[10];} mystruct;

int main()
{
  int a, b, c[10];
  mystruct s1, s2;
  s1.b[0] = &b;
  s1.a = &a;
  s1.c = &c;
  s2=s1;
  return(0);
}
