// to test memcpy with address_of arguments

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
	unsigned int	plop;
	char			plip;
	short			plup;
} My_Struct;

int main (int argc, char ** argv)
{
	unsigned int maxX = 1000;
	unsigned int maxY = 1000;

	My_Struct table1[maxY][maxX];
	int finalSum = 0;

	for (unsigned int y = 0; y < maxY; ++y)
	{
		for (unsigned int x = 0; x < maxX; ++x)
		{
			My_Struct base;
			base.plop = 999;
			base.plip = 5;
			base.plup = 512;

			memcpy( &(table1[y][x]), &base, sizeof(My_Struct));
		}
	}

	// display
	for (unsigned int y = 0; y < maxY; ++y)
	{
		for (unsigned int x = 0; x < maxX; ++x)
		{
			finalSum += table1[y][x].plop;
		}
	}

	printf("%d\n", finalSum);

	return 0;
}
