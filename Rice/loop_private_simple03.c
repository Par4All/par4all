/* Similar to loop_private_simple01 and 02.

   Make sure the declaration of z is not distributed in another
   loop... The statement is preserved but it is not linked by any
   dependence. We might need type dependence arcs to keep things
   together, because sizeof() and use of implicit declaration of
   structures are going to be distributed too.
 */

int main () {
  float a[10];
  int i;

  for (i = 0; i < 10; i++) {
    float z;
    z = 0.0;
    a[i] = z;
  }
  return 0;
}
