// to test chains for references to struct fields
// there should be no conflict between "s1.f1=0
// and the return statement
// however, the kill test is not yet sufficiently precise for
// simple effects, and the information is lost becasue of the predicate
// for convex effects

struct mystruct {
  int f1;
  int f2;
};

int main()
{
  struct mystruct s1;
  s1.f1 = 0;
  s1.f1 = 1;
  return s1.f1;
}
