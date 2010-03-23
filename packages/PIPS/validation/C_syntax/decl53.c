/* Management of "const" attribute */

/* pointer to a constant */
const int * x;
/* constant pointer */
int * const y;
/* another pointer to a constant */
const void * z;

// constant pointer to a constant: Well, this is forbidden!
// const * int const u;

void foo()
{
  x = 0;
  *y = 1;
  z = 0;
}
