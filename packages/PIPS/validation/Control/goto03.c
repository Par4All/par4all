/* Test a goto on the successor: */
void go_to() {
  int x = 6;
  goto below;
 below:
  x = 2;
}
