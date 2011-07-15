/* For IEF - Ter@ops */

#include <stdio.h>

typedef unsigned char byte;

void ReadPGMrow(FILE  *file, int width, byte  *line)
{
     fread(&(line[0]), sizeof(byte), width, file);
}


