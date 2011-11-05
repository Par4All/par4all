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
typedef void* P4A_accel_kernel;
P4A_accel_kernel main(unsigned int maxX, unsigned int maxY, unsigned int
macroX, unsigned int macroY, int *results, unsigned char *table1, unsigned char
*table2)
{
   // Loop nest P4A end
   if (macroY<=maxY-17&&macroX<=maxX-17) {
      int ret = Intermediate_function(macroX, macroY, maxX, maxY, table1, table2);
      *(results+macroX+macroY*maxX) = ret;                                        
   }
}

