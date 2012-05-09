void formal_parameter00(int *pi)
{
	int * q;
	int i;

	i = 0;
	q = pi;
	q++;
	pi = &i;
	return 0;
}
