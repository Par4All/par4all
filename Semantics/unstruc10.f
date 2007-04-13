      program unstruc10

C     Check behavior with subcycles (cycles are sure)

      real a(100, 100)

      read *, n

      if(n.lt.2) stop

      i = 1
      j = 1

 100  i = i + 1

 200  j = j + 1

      a(i,j) = 0.

      if(i.lt.n) go to 100
      if(j.lt.n) go to 200
      print *, i, j, n

      end
