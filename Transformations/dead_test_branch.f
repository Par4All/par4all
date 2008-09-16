      program dce
      logical l1, l2, l3, l4
      read *, l1, l2, l3, l4

      if (l1) then
         if (.FALSE.) goto 10
      else
         print *, 1
      endif
 10   continue

      if (l2) then
         if (.TRUE.) goto 20
      else
         print *, 2
      endif
 20   continue
      
      if (l3) then
         if (l4) goto 30
      else
         print *, 3
      endif
 30   continue

      end
