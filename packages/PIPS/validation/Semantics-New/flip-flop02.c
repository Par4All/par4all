/* Same as flip-flop01, but with a while loop to implement the for loop
 *
 * if(1) is used to gather more information about the analysis
 *
 * The internal loop is commented out to simplify debugging
 */

int main()
{
  int s[2][10];
  int i, j;
  int cur, next;
  cur = 0;
  next = 1;
  i = 1;
  while(i<10) {
	for (j = 1; j< 10; j++) {
	    s[cur][j] = s[next][j] + j;
	    s[next][j] = s[cur][j];
	  }
	cur = next;
	next = 1-next;
	i++;
    }
  return 0;
}
