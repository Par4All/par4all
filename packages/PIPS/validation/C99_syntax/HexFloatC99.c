#include <stdio.h>

main()
{
  float tmp1 = 1e+2;
  float tmp2 = 4e-3;
  float tmp3 = 0x1.f6297cp+1f;
  float tmp4 = 0x1.8f8b84p+3f;
  float tmp5 = 0x1.f6297cp-1f;
  float tmp6 = 0x1.8f8b84p-3f;
  printf("tmp1 = %f\n", tmp1);
  printf("tmp2 = %f\n", tmp2);
  printf("tmp3 = %f\n", tmp3);
  printf("tmp4 = %f\n", tmp4);
  printf("tmp5 = %f\n", tmp5);
  printf("tmp6 = %f\n", tmp6);
}
