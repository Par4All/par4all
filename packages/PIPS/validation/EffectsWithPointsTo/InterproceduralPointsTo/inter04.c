/* compute proper and cumulated effects for intra01
   compute proper and cumulated effects with points to for intra02
   compute proper and cumulated pointer effects for intra03
*/

void bar(int **pp, int **qq) {
  *pp = *qq;
  return;
}



void inter04(){
  int i = 0 , j = 1,  *p = &i, *q = &j, **pp = &p, **qq = &q ;
  
  bar(pp, qq);
  return;
}


