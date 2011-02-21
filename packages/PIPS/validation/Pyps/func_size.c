
void muladd(unsigned int n, float a[n], float b[n], float c[n])
{
    int i;
    for(i=0; i < n; i++)
        a[i] += b[i] + c[i];
}

int main()
{
	float a[100], b[100], c[100];
	muladd(100,a,b,c);
}
