// to test recursivity in eval_cell_with_points_to
int main(){
  int a, aa, *b, *bb, **c;

  a = 1;
  b = &a;
  c = &b;

  bb = *c;
  aa = **c;
  return 0;

}
