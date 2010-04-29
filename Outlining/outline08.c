#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
	int n;
	char * str;
	if(argc == 1 ) return 1;
	n=atoi(argv[1]);
	if( !(str=malloc(n*sizeof(char))) ) return 2;
	{
		int i;
		for(i=0;i<(n-1);i++)
			str[i]=('a'+(char)i);
		str[n-1]=0;
	}
	printf("%s\n",str);
	return 0;
}
