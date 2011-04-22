#include <unistd.h>


int benchmark(int n)
{
	int i,j;
	for (i=0;i<n;i++)
		j=j*i*n/2;
	return j;
}

int benchmark_sleep()
{
	usleep(100000);
}

int main()
{
	benchmark_sleep();
	return 0;
}
