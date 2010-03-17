      program cplx
      complex a, b, c
      integer l
      assign 10 to l
  10  a = (10,20)
      assign 20 to l
  20  b = (20,10)
  30  goto l, (10,20)
      end
