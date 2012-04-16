/* Privatization of global loop indices */

int __lv1;
int __lv2;

int main(int argc, char* argv[])
{
  int _u_a[3][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  _u_a[2][0]=7;
  _u_a[2][1]=8;
  _u_a[2][2]=9;
  double _u_r[3][3];
  for (__lv1=0;  __lv1<3;  __lv1++) {
    for (__lv2=0;  __lv2<3;  __lv2++) {
      _u_r[__lv1][__lv2] = sqrt(_u_a[__lv1][__lv2]);
    }
  }
  return 0;
}
