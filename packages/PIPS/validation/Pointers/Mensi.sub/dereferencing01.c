int main()
{
  struct MyStr{
    int ***p ;
  } ;
  struct MyStr *s, *r, t, v;
  int i = 0, j = 1, *p1 = &i, **p2 = &p1, *p3 = &j, **p4 = &p3;
  s = &t;
  s->p = &p2 ;
  r = &v;
  r->p = &p4 ;
  **(s->p) = **(r->p);

 return 0; 
}
