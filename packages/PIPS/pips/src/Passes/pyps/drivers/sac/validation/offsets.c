int f2(int off1, int off2, int w, int n, float r[n], float a[n], float b[n])
{
	int i,j,w,i1,i2,itmp1,itmp2;
	for (j = 0; j < w; j++)
	{
		itmp1 = off1+j*w;
		itmp2 = off2+j*w;
		for (i = 0; i < n; i++)
		{
			r[itmp1+i] = a[itmp1+i] + b[itmp2+i];
		}
	}
}

int f(int off1, int off2, int n, float r[n], float a[n], float b[n])
{
	int i,itmp1,itmp2;
	itmp1 = off1/n;
	itmp2 = off2/n;
	for (i = 0; i < n; i++)
	{
		r[itmp1+i] = a[itmp1+i] + b[itmp2+i];
	}
}
