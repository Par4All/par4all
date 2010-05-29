/*****************************************************************************
 * II - POINTER PARAMS
 * we may create a default location for b in order to be able
 * to handle points_to b[0] in intraprocedural analysis
 ****************************************************************************/
void dependence02( int *b ) {
  int *a;

  a = b; // a and b are aliased, a points_to b[0]

  *a = 0; // write effect on *a will be visible from callers
}
