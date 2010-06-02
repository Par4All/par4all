      subroutine double(b)
      integer b

 100  goto 200
 200  goto 100
      end
      
      subroutine simple

 100  goto 100
      end

      subroutine triple

 100  goto 300
 200  goto 100
 300  goto 200
      end

      subroutine if
      implicit integer (a-z)
 100  if (a .eq. 0) goto 100
      goto 100
      end

      subroutine goto double(b)
      implicit integer (a-z)

      b = -1

 100  goto 200
 200  goto 100
      
      asnieres = 22
      end
      
      subroutine if simple
      implicit integer (a-z)
      b = -2
 100  goto 100
      h 2 o = cent degres
      end

      subroutine subroutine triple
      implicit integer (a-z)
      b = - troye
 100  goto 300
 200  goto 100
 300  goto 200
      rien = a signaler
      end

      subroutine if de rien du tout
      implicit integer (a-z)

      sans if = trop facile
 100  if (a .eq. 0) goto 100
      goto 100
      il ne passera pas par la = vrai
      end

      subroutine des boucles tordues
      implicit integer (a-z)

      n = 2
      m = 10
      a = 1
      do i = 1, m, n
         a = a*a
         if (a .gt. 50) go to 100
      enddo
      
 100  if (i .ne. 11) then
         print *, 'Preconditions to be improved...'
      endif
      
      a = 1
      do i = 1, m, n
         a = a*2
         if (a .gt. 50) go to 200
      enddo
      
 200  print *, i, a
      
      a = 1
      m = -m
      n = -n
      do i = 1, m, n
         a = a*2
         if (a .gt. 50) go to 300
      enddo
      
 300  print *, i, a
      
      a = 1
      do i = 1, m, n
         a = a*a
         m = m + 3
         n = n + 3
         if (a .gt. 50) go to 400
      enddo
      
 400  print *, i, a
      
      end
