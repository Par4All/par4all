      program checktype

C     Goal: see if we know how to type check MAX, MAX0, AMAX1, DMAX1, AMAX0 and MAX1

      double precision x, y, z

      i = max0( i, j)

C     This one is wrong
      i = max0( i, j, 1.)

      end
