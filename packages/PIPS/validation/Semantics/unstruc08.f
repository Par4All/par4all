      program unstruc08

C     Check that certainly entered desugared loop might be handled

      real a(100)

      read *, n

      if(n.lt.1) stop

      do i = 1, n
         a(i) = 0.
         if(a(i).lt.a(i-1)) go to 200
      enddo

 200  continue
      print *, i, n

      end
