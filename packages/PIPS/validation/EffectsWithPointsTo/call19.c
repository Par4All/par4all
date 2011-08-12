typedef struct
{
  int data;
} something;

int bla(int * p)
{
  *p = 13;
  return 17;
}

int foo(int a)
{
  return a*3;
}

int call19(int * q)
{
  int i, k;
  something st;
  // bla(int *)
  i = bla(q);
  i += *q;
  i += bla(&k);
  i += k;
  i += bla(&st.data);
  i += st.data;
  // foo(int)
  i += foo(*q);
  i += foo(k);
  i += foo(st.data);
  return i;
}
