int formal_parameter01(int **pi)
{
	int ** q;
	int *i;

	i = 0;
	q = pi;
	pi = &i;
	return 0;
}
