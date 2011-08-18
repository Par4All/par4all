typedef int int32_t;

// ok if following line are exchanged
const int32_t k[1] = { 0 };
//const int k[] = 0;

int main(void)
{
  // ok if following line is uncommented
  //int i = 0;

  if(1 != 1) {
    // DEAD CODE TO BE REMOVED
    return 1;
  }

  return 0;
}
