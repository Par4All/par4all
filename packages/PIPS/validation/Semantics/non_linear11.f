      program non_linear11

C     Check the linearization of integer products

C     Both subexpressions are affine and equal

      i1 = (n+1)*(n+1)+1
      read *, i1

C     Both expressions are equal, i2 >=0

      i2 = (n*m)*(n*m)
      read *, i2

C     For intervals, there 9 different sign cases

C     ++++

      if(x.gt.0) then
         l1 = 1
         l2 = 2
      else
         l1 = 10
         l2 = 20
      endif

      k1 = l1*l2
      read *, k1

C     ++-+

      if(x.gt.0) then
         l1 = 1
         l2 = -2
      else
         l1 = 10
         l2 = 20
      endif

      k2 = l1*l2
      read *, k2

C     -+++

      if(x.gt.0) then
         l1 = -1
         l2 = 2
      else
         l1 = 10
         l2 = 20
      endif

      k3 = l1*l2
      read *, k3

C     ++--

      if(x.gt.0) then
         l1 = 1
         l2 = -20
      else
         l1 = 10
         l2 = 2
      endif

      k4 = l1*l2
      read *, k4

C     --++

      if(x.gt.0) then
         l1 = -10
         l2 = 2
      else
         l1 = -1
         l2 = 20
      endif

      k5 = l1*l2
      read *, k5

C     -+--

      if(x.gt.0) then
         l1 = -1
         l2 = -20
      else
         l1 = 10
         l2 = -2
      endif

      k6 = l1*l2
      read *, k6

C     ---+

      if(x.gt.0) then
         l1 = -10
         l2 = -2
      else
         l1 = -1
         l2 = 20
      endif

      k7 = l1*l2
      read *, k7

C     ----

      if(x.gt.0) then
         l1 = -10
         l2 = -20
      else
         l1 = -1
         l2 = -2
      endif

      k8 = l1*l2
      read *, k8

C     -+-+ (this case further splits down according to magnitudes)

      if(x.gt.0) then
         l1 = -1
         l2 = -2
      else
         l1 = 10
         l2 = 20
      endif

      k9 = l1*l2
      read *, k9

C     -+-+ Same as above, but with overflows

      if(x.gt.0) then
         l1 = -1 000 000
         l2 = -2 000 000
      else
         l1 = 10 000 000
         l2 = 20 000 000
      endif

      k10 = l1*l2
      read *, k10

      end
