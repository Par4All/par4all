// Check that when applying successively simplyfy_control,
// flatten_code, privatize_module and localize_declarations,
// the initialization of k is preserved

int main()
{
  int n = 2;
  for (int i=1; i<=n; i++) {
    int k = 1;
    while ( (k<10) ) {
      k = (k+1);
    }
  }

  return 0;
}
