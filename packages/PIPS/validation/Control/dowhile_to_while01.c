int main(int argc, char *argv[])
{
	int i =0;
	do {
		int j = 0;
		j = i++;
		printf("%d",j);
	} while( i < 10 );
	printf("\n");
	return 0;
}
