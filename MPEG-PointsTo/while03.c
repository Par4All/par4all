/* Make sure the side effect of the condition is taken into account
   when a while loop may be entered or not.

   End of trilogy while01 (never entered), 02 (always entered) and 03
   (maybe entered).
 */

#include <stdio.h>

main()
{
  int i = 0;
  int a[20];
  int n;

  scanf("%d", &n);

  while(i++<=n) {
    a[i] = i;
  }
  printf("%d\n", i);
}
