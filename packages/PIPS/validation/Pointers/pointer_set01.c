/* Bug due to the typedef: p is not recognized as a pointer if the
 * basic concrete type is not computed.
 */

typedef int * pointer;

void pointer_set(pointer p, int v)
{
  *p = v;
  return;
}
