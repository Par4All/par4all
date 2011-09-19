/* Make sure min and max are taken into account in for loop conditions */

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
