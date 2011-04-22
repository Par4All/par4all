/* Check the impact of loop distribution on declaration statements
 *
 * Here is the original case written by Pierre Villalon.
 *
 * It seems to work thanks to the initialization of z in the
 * declaration. This creates a dependence and works in spite of the
 * redundant write to z.
 */

int main () {
  float a[10];
  int i;

  for (i = 0; i < 10; i++) {
    float z = 0.0;
    z = 0.0;
    a[i] = z;
  }
  return 0;
}
