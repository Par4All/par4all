// check impact of parser on internal comments

//#include <stdio.h>

int comment08()
{
  int x[512];
  int i;
  int j;

  for(i = 0/*, j = 0*/; i<500;i +=128/*, j++*/)
    x[i] = 0;

  //printf("%d, %d\n", i, j);

 return 0;
}

main()\
{
  comment08();
}
