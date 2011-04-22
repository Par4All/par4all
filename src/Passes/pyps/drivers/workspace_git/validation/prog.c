#include <stdio.h>
#include "prog.h"

int main()
{
	int i=5;
	printf("Hi !\n");
	printf("%d\n", i);

	for (i=0; i<=10; i++)
		printf("%d\n",i);

	if (i==40)
		printf("hi\n");
	
	return 0;
}
