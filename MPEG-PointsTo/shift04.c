// Check analysis of shift operations (found in mpeg2enc)

#include <stdio.h>

int shift04(int i, int k)
{
  int j = i;
  int l = j>>2;
  int m = (-j)>>2;

  // Check a subset of particular cases: not found with the current
  // implementation of integer_right_shift_to_transformer
  if(j<0 && 2<= k && k <= 4)
    j = j >>k;
  else if(j>=0 && 2<= k && k <= 4 )
    j = j >>k;
  else
    j = j >>k;

  printf("j=%d, l=%d, m=%d\n", j, l, m);
  return j;
}

main()
{
  int i = 20;
  shift04(i, 2);
}
