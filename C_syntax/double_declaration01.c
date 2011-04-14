/* Legal double declaration of C
 *
 * See Ticket 430
 */

extern int c;
int c=1;

int main(void)
{
  return 0;
}
