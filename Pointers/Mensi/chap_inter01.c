void foo(int * pi, int * pj) {
  *pi=1, *pj=2;
  return;
}

int main() {
  struct {int * q;} s1, s2, *p, *r;
  int i, i1, i2;
  
  s1.q=&i1;
  s2.q=&i2;

  if(i>0)
    p = &s1, r = &s2;
  else
    p = &s2, r = &s1;

  foo(p->q, r->q);

  return 0;
}
