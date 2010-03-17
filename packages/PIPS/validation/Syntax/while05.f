      program while05

C     Check the controlizer

      real x(100)

      i = 100
      do 100 while(i.gt.1)
         if(x(i).gt.10.) go to 200
         x(i) = 0.
         i = i - 1
 100  enddo

 200  continue

      end
