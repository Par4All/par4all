// this is to validate the forward translation of struct of arrays regions

#include <string.h>
typedef struct  {
struct S1{
    struct S11 {
        float in[10][20][30];
    }S11;
}S1;

struct S2{
    struct S21 {
        float out[10][20][30];
    }S21;
}S2;

}SGlobal;

void foo(SGlobal * p_SG)
{
  for(int i=0; i<10; i++)
    for (int j =0; j<20; j++)
      for (int k = 0; k< 30; k++)
	p_SG->S1.S11.in[i][j][k] = i + j + k;
}

int main()
{
  SGlobal my_SG;

  foo(&my_SG);
  memcpy(&my_SG.S2.S21.out[0][0][0], &my_SG.S1.S11.in[0][0][0], 10*20*30*sizeof(float));
  return 0;
}
