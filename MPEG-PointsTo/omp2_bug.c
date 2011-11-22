/* Transformers in the second loop nest are wrong: 0==-1 is always
   returned when they are recomputed after privatization

   If the first loop nest is commented out, the problem
   disappears. But the second loop nest does not seem to be involved,
   or at least, there is a bug even with one loop nest.

   Initializations have been shortened to ease debugging
 */

int main ()
{
  int i = 0, j = 0;
  //float x = 2.12;
  float b[10][10];

  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      b[i][j] = 0.0;
    }
  }

  return 0;
}
