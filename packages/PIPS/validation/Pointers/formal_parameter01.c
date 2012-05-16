int formal_parameter01(int **pi)
{
	int ** q;
	int *i;
	int j;

	i = 0;
	q = pi;
	q++;
	pi = &i;
	*pi = &j;
	*q = &j;
	return 0;
}
