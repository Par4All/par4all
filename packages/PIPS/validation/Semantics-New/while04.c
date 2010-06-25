/* Make sure min and max are taken into account in while loop conditions */

#include <stdio.h>

#define min(x,y) ((x)<=(y)?(x):(y))

main()
{
  int i = 0;
  int a[20];
  int n;
  int m;
  int k = min(m,n);

  scanf("%d", &n);
  scanf("%d", &m);

  while(i++<=min(n,m)) {
    a[i] = i;
  }
  printf("%d\n", i);
}
