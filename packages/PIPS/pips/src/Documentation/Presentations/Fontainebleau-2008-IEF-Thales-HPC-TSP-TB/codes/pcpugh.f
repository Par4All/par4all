      program pcpugh
c producer-consumer algorithm 
      integer a, i, b, o1, o2
      real x
      data b, o1, o2 /0, 0, 0/
c Ask for the production amount
      read *, a
      if (a.lt.1) then
         stop
      endif
      i = a
      do k = 1, n
c        External event: 0,1,2
         read *, x
         if (x.gt.1.) then
c           Producer
            if (i.gt.0) then
               i = i - 1
               b = b + 1
            endif
         else if (x.lt.1.) then
c           Consumer 1
            if (b.gt.0) then
               o1 = o1 + 1
               b = b - 1
            endif
         else
c           Consumer 2
            if (b.gt.0) then
               o2 = o2 + 1
               b = b - 1
            endif
         endif
         if (i.eq.0 .and. b.eq.0) then
            print *,'The End',o1,o2,a
         endif
      enddo
      end
