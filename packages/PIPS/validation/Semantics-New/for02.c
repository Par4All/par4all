/* Check the impact of property SEMANTICS_KEEP_DO_LOOP_EXIT_CONDITION
   when set to false, which is not its default value */

#include <stdio.h>

main()
{
  int i = 0;
  int a[20];
  int n;

  scanf("%d", &n);

  for(i=0; i<=n; i++) {
    a[i] = i;
  }
  printf("%d\n", i);
}
