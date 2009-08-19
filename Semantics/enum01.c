//#include <stdio.h>

main()
{
  enum fleurs {rose=0x0001, marguerite=0, jasmin};
  enum legumes{carotte=rose+50, haricot};
  enum fleurs ma1, ma2, ma3;
  enum legumes mon1, mon2;
  int i;

  ma1 = rose;
  ma2 = marguerite;
  ma3 = jasmin;
  mon1 = carotte;
  mon2 = haricot;

  i = ma1+ma2+ma3+mon1+mon2;
}
