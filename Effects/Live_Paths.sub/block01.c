int main () {
  int result = 0;
  {
    int k = 0;
    result += k;
  }
  result ++;
  {
    int k = 1;
    result += k;
  }
  return result;
}
