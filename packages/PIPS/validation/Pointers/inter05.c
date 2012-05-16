/* compute proper and cumulated effects for intra01
   compute proper and cumulated effects with points to for intra02
   compute proper and cumulated pointer effects for intra03
*/

void bar(int ***ppp, int  ***qqq) {
  **ppp = **qqq;
  return;
}



void inter05(){
  int i = 0 , j = 1,  *p = &i, *q = &j, **pp = &p, **qq = &q, ***ppp = &pp, ***qqq = &qq;
  int k = 2, *r = &k;
  bar(ppp,qqq);
  return;
}


