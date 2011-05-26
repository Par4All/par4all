/*****************************************************************************
 * IV - GLOBAL VARIABLE
 ****************************************************************************/
int global_a;
void dependence04() {
  int *b;
  b = &global_a; // b points_to global_a

  *b = 0; // effect that may go out of current scope, b points on a global var
}
