int main()
{
  int a[10], *pa, b[10], *pb, res;

  pa = &a[0];
  pb = &b[0];

 begin:
  if (*pb + *pa > 0) goto end;

 middle1:
  pa ++;
  if (*pb + *pa > 0) goto end;

 middle2:
  pb ++;
  if (*pb + *pa > 0) goto end;
  else goto begin;

 end:
  res = *pa + *pb;

  return res;
}
