/* Same as flip-flop01, but with a k++ in the increment to avoid the
 * conversion to a DO loop.
 *
 * Property FOR_TO_WHILE_LOOP_IN_CONTROLIZER FALSE does not seem to
 * impact the conversion for the external for loop into a while loop.
 */

int main()
{
  int s[2][10];
  int i, j, k = 0;
  int cur, next;
  cur = 0;
  next = 1;
  for(i = 1; i<10 && k <100; i++, k++)
    {
    for (j = 1; j< 10; j++)
      {
	s[cur][j] = s[next][j] + j;
	s[next][j] = s[cur][j];
      }
    cur = next;
    next = 1-next;
    }
  return 0;
}
