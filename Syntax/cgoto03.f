      program cgoto03

C     Check handling of side effects

      icg0 = 1.

      i = 4

      goto(100,200,300,400), incr(i)
      j = 1
      go to 500
 100  continue
      j = 2
      go to 500
 200  continue
      j = 3
      go to 500
 300  continue
      j = 4
      go to 500
 400  continue
      j = 5

 500  continue
      print *, i, j
      end

      integer function incr(k)
      k = k + 1
      incr = k
      end

