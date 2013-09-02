      program test if 1
      integer a, b, j
      

C     Added to avoid empty summary preconditions
      if(.FALSE.) then
         call testif2(b)
         call testif3(b)
         call testif4(b)
         call testif5(b)
      endif


      j = 3
      

      if (j.eq. 5) goto 31
      print 9001, 'salut'
 9001 format ('j = ', i3)
 31   a = 9
      
      if (j.eq.3) then
         goto 1
      else
         goto 2
      endif
 2    print *, 'Faux 2'
 1    print *, 'Vrai 1'
      if (j.eq.4) then
         goto 3
      else
         print *, 'Vrai'
      endif
      
 3    print *, 'Faux 3'
      if (j.eq.3) then
         goto 3
      else
         goto 2
      endif
      
      print 9000, 'Jamais...'
 9000 format ('Jamais... j = ', i3)

      end
      subroutine test if 2(b)
      integer a, b, j
      
      j = 3
      
      b = j

c      if(.true.) then
      if (j.eq. 5) goto 31
      print *, 'salut'
 31   a = 9

c      endif
     

      end

      subroutine test if 3(b)
      integer a, b, j
      
      j = 3
      
      b = j
      a virer = j

      do i = 3, 10
         print *, 'une boucle !'
      enddo

c      if(.true.) then
      if (j.eq. 5) goto 31
      print *, 'salut'
 31   a = 9

c      endif
      a virer encore = b

      if (a .eq. i) then
         print *, 'vrai'
      else
         print *, 'faux'
      endif

 100  goto 200
 200  goto 100

      print *, 'jamais'
      end

      subroutine test if 4(b)
      integer a, b, j
      real x(10), y(10)
      
      j = 5
      b = j
      x(1) = y(2)
      y(2) = b
     
      end

      subroutine test if 5(b)
c test the propagation of control usefulness through the predecesors      
      integer a, b, j
      real x(10), y(10)
      
      j = 5
      goto 100
 100  b = j
      if (a .ne. 0) goto 200
      print *, 'Well...'
 200  to be discarded = 1
      yacc too = 2
      x(1) = y(2)
      y(2) = b
     
      end

