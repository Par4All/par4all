      program unstruc09

C     Check behavior with subcycles

      real a(100, 100)

      read *, n

      i = 1
      j = 1

 100  i = i + 1

 200  j = j + 1

      a(i,j) = 0.

      if(i.lt.n) go to 100
      if(j.lt.n) go to 200
      print *, i, j, n

      end
