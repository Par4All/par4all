/* Use global variables
 *
 * Since the analysis is bottom-up, the global variables are not
 * initialized and a formal context is used instead.
 */

typedef struct two_fields{int one; int two[10];} tf_t;

int i[10];
tf_t s;
int * pi = &i[0];
tf_t *q = &s;

void call03()
{
  *pi = 1;
  pi++;
  *pi = 2;
  q->one = 1;
  q->two[4] = 2;
  return;
}

int main()
{
  call03();
  return 0;
}
