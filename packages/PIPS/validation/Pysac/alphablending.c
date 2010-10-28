#include <stdio.h>
#include <stdlib.h>

extern void alphablending(unsigned int n, float src1[n], float src2[n], float result[n], float alpha);

int main(int argc, char **argv)
{
	unsigned int n;
	float *src1, *src2, *result;
	float alpha;
	n = atoi(argv[1]);
	printf(">>>> %d <<<<\n", n);
	alpha = 0.7;

#define xmalloc(p)							\
	do {								\
		if (posix_memalign((void **) &p, 32, n * sizeof(float))) \
			return 3;					\
	} while(0);

	xmalloc(src1);
	xmalloc(src2);
	xmalloc(result);

	alphablending(n, src1, src2, result, alpha);

	return (int) (result[0] + result[n - 1]);
}
void alphablending(unsigned int n, float src1[n], float src2[n], float result[n], float alpha)
{
    unsigned int i;
    for(i=0;i<n;i++)
        result[i]=alpha*src1[i]+(1-alpha)*src2[i];
}
