
!
! inegalites sur les flottants
!

      program inegfloat03
      real r

! unknown

      read *, r

! float ge le

      if (r.ge.0.0.and.r.le.1.0) then
         print *, 'float ge le float'
         print *, r
      endif

! known-in

      r = 0.5

! float ge le

      if (r.ge.0.0.and.r.le.1.0) then
         print *, 'float ge le float'
         print *, r
      endif

      print *, i, r, d

      end
