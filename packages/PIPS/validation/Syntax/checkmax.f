      program checkmax

C     Goal: see if we know how to type check MAX, MAX0, AMAX1, DMAX1, AMAX0 and MAX1

      double precision x, y, z

      i = max0( i, j)

      i = max0( i, j+1)

C     This one is wrong
      i = max0( i, j, 1.)

      i = max( i, j, 1., x)

      u = amax1(v, w)

C     This one is wrong
      u = amax1(v, w, 1)

C     This one is correct because the integer 1 should be converted
      u = amax1(v, w, w+1)

      x = dmax1(y, z)

      x = dmax1(y, z, z+1., 1.D0)

      u = amax0(i, j, k, l, m, n)

C     This one is wrong because +1. is going to generate a real argument
      u = amax0(i, j, k, l, m, n+1.)

      i = max1(u, v, w)

C     This one is wrong
      i = max1(u, v, w, 1)

C     Overloaded operator...
      i = max(i, u, w)

      end
