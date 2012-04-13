/* Check recursive stub generation */

int formal_parameter03(int ppp[10][10][10])
{
	int *q = ppp[0][0];
	int **qq = ppp[0];
	int ***qqq = ppp;
	return 0;
}
