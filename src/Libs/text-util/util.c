#include <stdio.h>
#include <string.h>

char *i2a(i)
int i;
{
    static char buffer[32];
    sprintf(buffer, "%d", i);
    return(strdup(buffer));
}    

char *f2a(f)
float f;
{
    static char buffer[32];
    sprintf(buffer, "%f", f);
    return(strdup(buffer));
}    
