#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
    int j,n;
    char * str=0;
    if(argc == 1 ) return 1;
l0:for(j=0;j<1;j++)
   {
       n=atoi(argv[1]);
       if( (str=malloc(n*sizeof(char))) )
       {
           int i;
           for(i=0;i<(n-1);i++)
               str[i]=('a'+(char)i);
           str[n-1]=0;
       }
   }
   if(str)
       printf("%s\n",str);
   return 0;
}
