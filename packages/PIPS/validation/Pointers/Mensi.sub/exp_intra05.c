int main()
{
int *p;   // S1
 p = (void*) 0 ;     // S2
*p = 1 ; // S3
 return 0;
}
