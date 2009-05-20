void foo(int a[30][20][10], int b[20][10],int c[10],int d) 
{
	a[20][10][0]=b[10][0];
	c[0]=d;
}

void bar(int c[30][20][10])
{
	foo(c[0],c[0][1],c[0][2],c[0][2][4]);
}
