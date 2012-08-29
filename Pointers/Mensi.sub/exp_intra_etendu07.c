int main()
{
 char **p, *q, *r, *s, i, j, k;
 i = j = k = 'c'; 
 q = &i;
 r = &j; 
 if(i=='h')
  p = &q;
 else
  p = &r;
  s = &k;
  *p = s;
}
