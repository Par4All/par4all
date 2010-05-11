/* Test kernel_load_store on a scalar modification.

   Assume that only the pointed scalar is touched, since it is often the
   case for generated code given to kernel_load_store
*/

enum { N = 100 };

void change(int *i, double array[N]) {
  int k;

  (*i)++;

  for(k = 0; k < N; k++)
    array[k] = 0;
}

void give() {
  int j = 3;
  double array[N];

  change(&j, array);
}
