#include <math.h>

float expdev(idum)
int *idum;
{
	float ran1();

	return -log(ran1(idum));
}
