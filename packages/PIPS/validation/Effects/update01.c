typedef struct{
  int * pi; } wt;

void foo(wt * s)
{
  *(s)->pi++ = 0;
}
