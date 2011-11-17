struct complex {
  int re;
  int im;
};

struct complex i() {
 struct complex _i = { 0, 1 };
 return _i;
}

main() {
 int im = i().im;
 return im;
}
