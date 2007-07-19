/* Checks the memory allocation offsets for local dynamic variables of dynamic size */

extern int n;
extern int m;
void fcompat()
{
  int a[n];
  int (*p)[n+1];
  int c[m];

}
