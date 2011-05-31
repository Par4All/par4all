#include <math.h>

/* flatten_code should not spoil the code with useless definitions

   In this test case, we expect only one Pips generated varaible : F_212.
   Not four of them :  double F_212_0, F_212_1, F_212_2, F_212_3;
*/

void flatten_code16() 
{
  double U[2];
  double _return391;
  {
    {
      {
	{
	  double F_212;
	  F_212 = U[1];
	  _return391 = F_212;
	}
      }
    }
  }
}


int main(int argc, char **argv)
{
  flatten_code16();
}
