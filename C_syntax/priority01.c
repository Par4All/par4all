int priority01() {
  char a, b, c;
  //a = 1;
  //b = 2;
  /* Should be prettyprint as "(c = a + b) != '\n'"
     and not "c = a + b != '\n'" */
  return (c = a + b) != '\n';
}
