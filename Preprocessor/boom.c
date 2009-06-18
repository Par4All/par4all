 void f(char* s) {
}

void boom() {
  f("\\");
  f("abc");
}

int main () {
  boom ();
  return 0;
}
