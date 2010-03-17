void miam(void) {
 bara:
 gwin:
  goto bara;
  // This goto should not disappear:
  goto gwin;

 gwen:
  {
  ruz:
    goto gwen;
  }
}
