      program unstr
c     To verify the test is not restructured
c     (the entry node has a fan in of 2)
      integer a, b

      
c     10 j = 3
 10   j = 3
      goto 20
c     20 a = 2
 20   a = 2

c     if:
      if (b .eq. 0) goto 20
      goto 10
      

      end
