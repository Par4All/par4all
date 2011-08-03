      program power03

C     Excerpt from turb3d

      read *, m

      if(m.lt.5.or.m.gt.6) stop

      n = 2**m
      m1 = min(m/2, 2)
      m2 = m-m1
      n2 = 2**m1
      n1 = 2**m2

c     Get rid of auxiliary variables
      read *, m1, m2

      print *, i, n, n1, n2

      end
