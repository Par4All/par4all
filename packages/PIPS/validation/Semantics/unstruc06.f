      program unstruc06

C     Check that desugared loop might be handled

      real a(100)

      read *, n

      i = 1
 100  continue
      if(i.gt.n) go to 200
      a(i) = 0
      i = i + 1
      go to 100

 200  continue
      print *, i, n

      end
