//#include <stdio.h>

int main()
{
  enum fleurs {rose=0x0001, marguerite=0, jasmin};
  enum legumes{carotte=rose+50, haricot};
  enum fleurs ma1, ma2, ma3;
  enum legumes mon1, mon2;
  int i, a1, a2, a3, on1, on2, j;

  ma1 = rose;
  ma2 = marguerite;
  ma3 = jasmin;
  mon1 = carotte;
  mon2 = haricot;

  i = ma1+ma2+ma3+mon1+mon2;

  a1 = rose;
  a2 = marguerite;
  a3 = jasmin;
  on1 = carotte;
  on2 = haricot;

  j = a1+a2+a3+on1+on2;

  // printf("rose=%d, margueritte=%d, jasmin=%d\n", ma1, ma2, ma3);
  //printf("carotte=%d, haricot=%d\n", mon1, mon2);
  return 0;
}
