// After coarse_grain_parallelisation, expression are normalized
// Flatten code used to replace entities in expression without updating the normalized field
// Thus we didn't manage to fuse loops


int main(int argc, char *argv[])
{
   int a[10],b[10];
   {
      int i;
      for(i = 0; i <= 10; i += 1)
        a[i] = 0;
   }
   {
      int i;
      for(i = 0; i <= 10; i += 1)
        b[i] = a[i];
   }
 }
 
