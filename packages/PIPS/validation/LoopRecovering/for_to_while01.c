int main() {
  int a[100];
  int i, j;

  /* Test we can deal with comments and label in while() { } generation. */
  for(i = 2; i <= 50; i++)
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  for(i = 2; i <= 50; i++)
    /* A comment */
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  for(i = 2; i <= 50; i++)
  a_label:
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  for(i = 2; i <= 50; i++)
  a_label_before_comment:
    /* A comment */
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  for(i = 2; i <= 50; i++)
    /* A comment */
  a_label_after_comment:
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  for(i = 2; i <= 50; i++)
    /* A comment */
  a_label_between:
    /* Another comment */
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  return 0;
}
