/* Check that dereferencement errors are spotted. */

int main()
{
   int *p = (void *) 0, i;

   if (*p)
      p = (void *) 0;
   else
      p = &i;
   return 0;
}
