#include <stdio.h>

void getimage(filename)

char *filename;

{	unsigned char c;
	FILE *fp;

	fp=fopen(filename,"r");
	do
	{        while((c=fgetc(fp))!='\n');;
	}while((c=fgetc(fp))=='#');            /* Skip comment lines */

	fclose(fp);
}
