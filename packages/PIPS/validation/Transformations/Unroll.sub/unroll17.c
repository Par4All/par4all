/* Check full unrolling of a loop containing a typedef statement */

void unroll17()
{
  int i, j[10];
 bar:for(i=0;i<10;i++) {
    typedef int mtype;
    mtype x = i;
    j[0]=x;
  }
}
