/* Model Checking Sequential Software Programs
   ACM Trans. on Design Automation of Electronic Systems
   26 pages, article 10, 2009, v. 14, n. 1*/

void bar() {
  int x = 3;
  int y = x - 3;
  while(x<=4) {
    y++;
    x = foo(x);
  }
  y = foo(y);
}

int foo(int s)
{
  int t = s + 2;

  if(t>6)
    t -= 3;
  else
    t--;

  return t;
}
