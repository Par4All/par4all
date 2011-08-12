#include <stdio.h>
void lonesome_cowboy()
{
  int i,a[12];
  for(i=0;i<12;i++)a[i]=1;
  for(i=1;i<11;i++)
    a[i]=0;
 isolate:
  for(i=0;i<12;i++) printf("%d",a[i]);
}
main()
{
    lonesome_cowboy();
    return 0;
}
