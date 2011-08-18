// array a overflow?

#define size 10

void privatize_local_variable()
{
   int j;
   float sum;
   float a[10];
   for (j=0; j<size; j++) {
     sum+=a[10];
   }
}

void privatize_function_params(float sum )
{
   int j;
   float a[10];
   for (j=0; j<size; j++) {
     sum+=a[10];
   }
}


