      program fabien
      implicit integer(a-z)

c     Problem with expression normalization

      integer n, m, i
cfirst      parameter(n=101, m=24, l1=min(n,m), l2=max(n,m), l3=m**2, 
cfirst     &     l4=mod(n,m), l5=n/m)

c     Alternative to PARAMETER (partial evaluation)
csecond      n = 101
csecond      m = 24
csecond      l1 = min(n,m)
csecond      l2=max(n,m)
csecond      l3=m**2
csecond      l4=mod(n,m)
csecond      l5=n/m

c     Check that parameters are properly evaluated (eval.c)

      j1 = l1
      j2 = l2
      j3 = l3
      j4 = l4
      j5 = l5
      print *, j1, j2, j3, j4, j5

c     Check expression normalization or partial evaluation

      i1 = min(n, m)
      i1 = min0(n, m)
      i2 = max(n, m)
      i2 = max0(n, m)
      i3 = m**2
      i4 = mod(n, m)
      i5 = n/m
      print *, i1, i2, i3, i4, i5

      end
