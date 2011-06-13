// The comma expressions inter with the controlizer

#include <stdio.h>

int main() {
  int i;

  for(i=0;i<10;printf("%d\n",i), i++)
    ;
}
