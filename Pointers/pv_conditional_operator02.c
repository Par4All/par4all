// conditional operator and structures

// FI: fields b and c should be initialized in s2 and s3 before they
// are assigned to s1.

typedef struct {int *a; int *b[10]; int (*c)[10];} mystruct;

int main()
{
  mystruct s1, s2, s3;
  int b = 1, c = 2, d;

  s2.a = &b;
  s3.a = &c;
  s1 = (c == d)? s2 : s3;
  return(0);
}
