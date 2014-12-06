/* Check handling of typedef conflicts */

void typedef02() {
  typedef char * mytype;
  int i;
  {
    int i;
    typedef int mytype;
    mytype j;
  }
  {
    int i;
    typedef double mytype;
    mytype j;
  }
}
