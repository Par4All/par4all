#include <stdio.h>

void function(  
  int nWidth, 
  int nHeight, 
  char array[nHeight][nWidth])
{
  for (int i = 0; i < nHeight; i++)
    for (int j = 0; j < nWidth; j++)
      array[i][j] = 0;

} 

int main( int argc, char ** argv )
{
  int x = 666;
  int y = 123;

  char bigArray[ x * y ];					// big flat arrray
  
  char (*simpleArray)[y][x] = (char (*)[y][x]) bigArray; 
  
  function(
  	x, 
  	y,  
  	*((char (*)[y][x]) bigArray)
  	//ok: *simpleArray
  );

  return 0;
}



