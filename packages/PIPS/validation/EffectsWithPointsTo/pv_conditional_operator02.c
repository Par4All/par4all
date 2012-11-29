// conditional operator and structures

typedef struct {int *a; int *b[10]; int (*c)[10];} mystruct;

int main()
{
  mystruct s1, s2, s3;
  int b = 1, c = 2, d = 3;

  s2.a = &b;
  s3.a = &c;
  s1 = (c == d)? s2 : s3;
  return(0);
}
