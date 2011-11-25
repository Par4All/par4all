
void comment_before_test(int a, int b)
{
  //Shouldn't be lost
  // but use to be because of the ?: syntax in the test
  for (a<((b<0)?1:2);;) {
  }
}

