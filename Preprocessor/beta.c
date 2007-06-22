#include <math.h>

float beta(z,w)
float z,w;
{
	float gammln();

	return exp(gammln(z)+gammln(w)-gammln(z+w));
}
