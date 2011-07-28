struct mystruct {
  int f1;
  int f2;
};

int main()
{
  struct mystruct s1;
  s1.f1 = 0;
  s1.f2 = 1;
  return s1.f1+s1.f2;
}
