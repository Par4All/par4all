/* Simplified version of decl42.c: line numbering wrong for first
   exit() call statement in decl42.c.


   Not replicated here without the include.
 */
void * safe_malloc(int n)
{
  long i;

  if(!i) {
    void exit(int);
    exit(2);
    exit(4);
  }
  else
    return (void *) i;
}
