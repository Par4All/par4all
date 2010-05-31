/* Bug: test on non NULL pointer not supported by semantics.

   There might be more work on this when pointers are analyzed by
   semantics.
*/

void transformer01 (float *src) {
  /* Check a NULL pointer */
  if(src)
    ;
}
