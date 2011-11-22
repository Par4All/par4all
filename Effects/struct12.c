struct complex {
  int re;
  int im;
};

struct complex i() {
 struct complex _i = { 0, 1 };
 return _i;
}

main() {
  int im1 = ((struct complex) {0, 1}).im;
  int im2 = i().im;
 return im1 + im2;
}
