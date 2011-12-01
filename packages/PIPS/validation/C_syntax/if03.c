// See trac #601, the second part

int main(int argc, char* argv[])
{
  /*  t82.sce - testing elseif */
  int _u_a = 3;
  int _u_b = 5;
  int _u_c = 4;
  if ((_u_a>_u_b)) {
    if (_u_b) {
      call("a=b");
    }
  } else { 
    if ((_u_a<_u_b)) {
      if ((_u_a<_u_c)) {
        call("a<c<b");
      }
    }
  }
}

