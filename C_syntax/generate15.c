// Check the generation for unknown functional parameters

// The unknown return type must be printed as "int"

// It is even better to check that the source code submitted to PIPS
// is correct, using gcc or another C compiler

// Here, the C code is wrong, but it is on purpose,,,

//int foo(int n)
//{
//  return n;
//}

void generate15()
{
  //  int i = 0;

  func(/*i,*/ foo);
}
