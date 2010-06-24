/* Make sure min and max are taken into account in for/do loop conditions */

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

  for(i=0; i<=min(n,m); i++) {
    a[i] = i;
  }
  printf("%d\n", i);
}
