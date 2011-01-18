// The final semicolon combined with the previous pragma made Pips crash
void pragma(unsigned int n, float *res, float *a, float *b)
{
	unsigned int i;
#pragma start
	for (i=0;i<n;i++) {
		res[i]=a[i]+b[i];
	}
#pragma end
	;
}
