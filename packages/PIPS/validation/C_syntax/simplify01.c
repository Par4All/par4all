// Check that initialization j=i++, just after declaration and
// initialization of i does not kill the parser

#include <stdio.h>

int main(int argc, char* argv[])
{
	int i=0,j=i++,k=0;
	for(i=0;i<10;i++)
		for(j=0;j<10;j++)
			k+=i*j;
	printf("%d\n",k);
	return 0;
}
