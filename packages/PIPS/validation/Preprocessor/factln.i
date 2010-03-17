float factln(n)
int n;
{
	static float a[101];
	float gammln();
	void nrerror();

	if (n < 0) nrerror("Negative factorial in routine FACTLN");
	if (n <= 1) return 0.0;
	if (n <= 100) return a[n] ? a[n] : (a[n]=gammln(n+1.0));
	else return gammln(n+1.0);
}
