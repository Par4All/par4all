      program nga04

C     Conditions for array bound checking: you want N<=33 and M <= 33
C     before the print statement. This is not true if the two loops are
C     not entered. The full postcondition is:

C     n <= 0 OR ( n >= 1 AND m <=0 ) OR ( 1 <= n <= 33 AND 1 <= m <= 33 )

C     To obtain useful conditions for array bound checking, you have to
C     enter the i loop, you have to know that n is positive. Maybe we
C     should sometimes use the array bound declaration and declare that
C     REAL A(N) implies N>=1 although it is not in the FOrtran standard

      real a(33,33)

      read *, n, m

c      if(n.lt.5) stop
c      if(m.lt.5) stop

      do i = 1, n
         do j = 1, m
            if(i.gt.33) stop
            if(j.gt.33) stop
            a(i,j) = 0.
         enddo

         print *, a, n, m

      enddo

      print *, a, n, m

      end
