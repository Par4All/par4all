//same that side_effect01 but with lhs an rhs different type 
// and lhs type not analyze
// but the side effect must take place
// we want i=1 at the end

int main()
{
  int i=0;
  float x= (i=1)+2.;
  
  return 0;
}

