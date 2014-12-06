/* Damien Masse, Policy-Iteration-Based Conditional Termination and
 * Ranking Functions, VMCAI 2014, pp. 453-471
 *
 * Figure 6, p. 462
 *
 * Useful to demonstrate usefulness of while-if conversion
 */

void masse_vmcai_2014_06(int x, int y)
{
  while(x>=0) {
    x = x + y;
    if(y>=0) y--;
    // Body postcondition:
    ;
  }
}

void masse_vmcai_2014_06_transformed(int x, int y)
{
  while(x>=0) {
    while(x>=0 && y>=0) 
      x = x + y, y--;
    while(x>=0 && y<0)
      x = x + y;
    // Body postcondition:
    ;
  }
}
