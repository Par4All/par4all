int __lv1;
int __lv2;

int main(int argc, char *argv[])
{
   int _u_i;

   for(_u_i = 1; _u_i <= 10; _u_i += 1) {
      double _tmpxx0[10][10];
      for(__lv1 = 0; __lv1 <= 9; __lv1 += 1)
         for(__lv2 = 0; __lv2 <= 9; __lv2 += 1)
            _tmpxx0[__lv1][__lv2] = (double) 1.0;
      double _u_a[10][10];
      for(__lv1 = 0; __lv1 <= 9; __lv1 += 1)
         for(__lv2 = 0; __lv2 <= 9; __lv2 += 1)
            _u_a[__lv1][__lv2] = _tmpxx0[__lv1][__lv2]*(double) _u_i;
   }

}
