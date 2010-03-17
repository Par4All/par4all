      subroutine lower0(n)

C     Check integration for negative lower bounds

C     Note: "-2" represents the unary operator "-"
C     applied to the positive integer constant "2"

      do 30 i=-2,n
         t = t + 1.
 30   continue

      do 50 i=-1,n
         t = t + 1.
 50   continue

      do 10 i=0,n
         t = t + 1.
 10   continue

      do 20 i=1,n
         t = t + 1.
 20   continue

      do 40 i=2,n
         t = t + 1.
 40   continue

      end

      
