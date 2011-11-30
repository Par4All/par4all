
int main () {
  int m = 0;
  float y = .5;
  int result = 0;
  for (m = 0; m < 10; m++) {
    result += m;
    y = cos (y);
  }
  result += y * 10.0;
  return result;
}
