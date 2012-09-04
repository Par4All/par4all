/* A variation of conditional01.c
 *
 * The result for r is not false, but it is not normalized: r points
 * to "a" or any character string....
 */

const char * const x = "a";
const char * const y = "b";
const char * const z = "c";

char * conditional04(int i)
{
  char * p[3];
  p[0] = x;
  p[1] = y;
  p[2] = z;
  char * r = (i<0||i>2)? p[0]:p[i];
  return r;
}
