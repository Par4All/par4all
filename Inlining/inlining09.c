void foo(int * a) 
{
	*a=0;
}

void bar(int **a)
{
	foo(*a);
}
