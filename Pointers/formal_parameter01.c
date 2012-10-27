/* pi is written and ends up pointing to nowhere */

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

int main()
{
  int i, *ip, **aipp;
  ip = &i;
  aipp = &ip;
  formal_parameter01(aipp);
  return 0;
}
