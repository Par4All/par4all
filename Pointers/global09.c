/* Check that global initializations are taken into account when the
 * const qualifier is used 
 */

int a[10];
int * const pa = &a[0];
int * qa = &a[0];

int global09()
{
  int *p = pa;
  qa++;
  return *p;
}
