/* Combine an involutive part with an evolutive one
 *
 * Use to check delayed differentiation
 */

int main()
{
  int c = 0;
  int i = 0;
  int j = 0;

  while(j<=10) {
    if(c==0)
      i++, j++;
    else {
      c=0;
      i--, j++;
    }
  }
  return;
}
