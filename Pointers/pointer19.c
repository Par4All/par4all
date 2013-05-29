/* Excerpt of pointer14: just the "foo" function 
 *
 * Because the formal parameter is a an array of pointers, an
 * additional dimention is added for the sink:_t_1[3][4][*][5][6][7]
 * for a total of 6 dimensions. The extra dimension is used if pointer
 * arithmetic is applied to an array element.
 *
 * Since each element is a pointer, their target, _t_1_1, must be a
 * seven dimension array, again to allow pointer arithmetic.
 *
 * A bug is detected with POINTS_TO_DEBUG_LEVEL=1 to check typing.
 */

// array of pointers towards arrays

void foo(double * (*t[3][4])[5][6][7])
{
  *(*(t[2][3]))[1][2][3] = 2.5;
  return;
}
