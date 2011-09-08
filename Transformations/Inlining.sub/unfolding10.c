#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16


int Intermediate_function( 
	unsigned int startX, 
	unsigned int startY, 
	unsigned int width, 
	unsigned int height, 
	unsigned char array1[height][width], 
	unsigned char array2[height][width])
{
	int sum = 0;
	for (unsigned int y = 0; y < height; ++y)
	{
		for (unsigned int x = 0; x < width; ++x)
		{
			sum += (array1[y + startY][x + startX] - array2[y + startY][x + startX]);
		}
	}
	return sum;
}


int main (int argc, char ** argv)
{
	unsigned int maxX = 1000;
	unsigned int maxY = 1000;

	unsigned char table1[maxY][maxX];
	unsigned char table2[maxY][maxX];
	int results[maxY][maxX];
	int finalSum = 0;

	//init
	for (unsigned int y = 0; y < maxY; ++y)
	{
		for (unsigned int x = 0; x < maxX; ++x)
		{
			table1[y][x] = (unsigned char)((rand() * 256) / RAND_MAX);
			table2[y][x] = (unsigned char)((rand() * 256) / RAND_MAX);
			results[y][x] = 0;
		}
	}
	

	for (unsigned int macroY = 0; macroY < maxY - BLOCK_SIZE; ++macroY)
	{
#pragma pips outline
		for (unsigned int macroX = 0; macroX < maxX - BLOCK_SIZE; ++macroX)
		{
			int ret = Intermediate_function(macroX, macroY, maxX, maxY, table1, table2);
			results[macroY][macroX] = ret;
		}
	}

	// display 
	for (unsigned int y = 0; y < maxY; ++y)
	{
		for (unsigned int x = 0; x < maxX; ++x)
		{
			finalSum += results[y][x];
		}
	}

	printf("%d\n", finalSum);

	return 0;
}
