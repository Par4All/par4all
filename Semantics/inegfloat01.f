!
! inegalites sur les flottants
!

      program inegfloat01
      integer i
      real r
      double precision d

! unknown

      read *, i
      read *, r
      read *, d

! int gt lt

      if (i.gt.0) then
         print *, 'int gt int'
         print *, i
      endif

      if (i.gt.0.and.i.lt.1) then
         print *, 'int gt lt int'
         print *, i
      endif

      if (i.gt.0.0.and.i.lt.1.0) then
         print *, 'int gt lt real'
         print *, i
      endif

      if (i.gt.0.0D0.and.i.lt.1.0D0) then
         print *, 'int gt lt double'
         print *, i
      endif

      if (i.gt.0.0E0.and.i.lt.1.0E0) then
         print *, 'int gt lt double'
         print *, i
      endif

! int ge le

      if (i.ge.0.and.i.le.1) then
         print *, 'int ge le int'
         print *, i
      endif

      if (i.ge.0.0.and.i.le.1.0) then
         print *, 'int ge le real'
         print *, i
      endif

      if (i.ge.0.0D0.and.i.le.1.0D0) then
         print *, 'int ge le double'
         print *, i
      endif

      if (i.ge.0.0E0.and.i.le.1.0E0) then
         print *, 'int ge le float'
         print *, i
      endif

! float gt lt

      if (r.gt.0.and.r.lt.1) then
         print *, 'float gt lt int'
         print *, r
      endif

      if (r.gt.0.0.and.r.lt.1.0) then
         print *, 'float gt lt float'
         print *, r
      endif

      if (r.gt.0.0D0.and.r.lt.1.0D0) then
         print *, 'float gt lt double'
         print *, r
      endif

      if (r.gt.0.0E0.and.r.lt.1.0E0) then
         print *, 'float gt lt real'
         print *, r
      endif

! combines

      if (2*r.gt.0.and.2.*r.lt.1) then
         print *, 'float gt lt int int comb'
         print *, r
      endif

      if (2.0*r.gt.0.and.2.0*r.lt.1) then
         print *, 'float gt lt int float comb'
         print *, r
      endif

      if (2*r.gt.0.0.and.2*r.lt.1.0) then
         print *, 'float gt lt float int comb'
         print *, r
      endif

      if (2.0*r.gt.0.0.and.2.0*r.lt.1.0) then
         print *, 'float gt lt float float comb'
         print *, r
      endif

! float ge le

      if (r.ge.0.and.r.le.1) then
         print *, 'float ge le int'
         print *, r
      endif

      if (r.ge.0.0.and.r.le.1.0) then
         print *, 'float ge le real'
         print *, r
      endif

      if (r.ge.0.0D0.and.r.le.1.0D0) then
         print *, 'float ge le double'
         print *, r
      endif

      if (r.ge.0.0E0.and.r.le.1.0E0) then
         print *, 'float ge le float'
         print *, r
      endif

! double gt lt

      if (d.gt.0.and.d.lt.1) then
         print *, 'double gt lt int'
         print *, d
      endif

      if (d.gt.0.0.and.d.lt.1.0) then
         print *, 'double gt lt float'
         print *, d
      endif

      if (d.gt.0.0D0.and.d.lt.1.0D0) then
         print *, 'double gt lt double'
         print *, d
      endif

      if (d.gt.0.0E0.and.d.lt.1.0E0) then
         print  *,'double gt lt real'
         print *, d
      endif

! double ge le

      if (d.ge.0.and.d.le.1) then
         print *, 'double ge le int'
         print *, d
      endif

      if (d.ge.0.0.and.d.le.1.0) then
         print *, 'double ge le float'
         print *, d
      endif

      if (d.ge.0.0D0.and.d.le.1.0D0) then
         print  *,'double ge le double'
         print *, d
      endif

      if (d.ge.0.0E0.and.d.le.1.0E0) then
         print  *,'double ge le real'
         print *, d
      endif

! known-in

      i = 0
      r = 0.5
      d = 0.5D0

! int gt lt

      if (i.gt.0) then
         print *, 'int gt int'
         print *, i
      endif

      if (i.gt.0.and.i.lt.1) then
         print *, 'int gt lt int'
         print *, i
      endif

      if (i.gt.0.0.and.i.lt.1.0) then
         print *, 'int gt lt real'
         print *, i
      endif

      if (i.gt.0.0D0.and.i.lt.1.0D0) then
         print *, 'int gt lt double'
         print *, i
      endif

      if (i.gt.0.0E0.and.i.lt.1.0E0) then
         print *, 'int gt lt double'
         print *, i
      endif

! int ge le

      if (i.ge.0.and.i.le.1) then
         print *, 'int ge le int'
         print *, i
      endif

      if (i.ge.0.0.and.i.le.1.0) then
         print *, 'int ge le real'
         print *, i
      endif

      if (i.ge.0.0D0.and.i.le.1.0D0) then
         print *, 'int ge le double'
         print *, i
      endif

      if (i.ge.0.0E0.and.i.le.1.0E0) then
         print *, 'int ge le float'
         print *, i
      endif

! float gt lt

      if (r.gt.0.and.r.lt.1) then
         print *, 'float gt lt int'
         print *, r
      endif

      if (r.gt.0.0.and.r.lt.1.0) then
         print *, 'float gt lt float'
         print *, r
      endif

      if (r.gt.0.0D0.and.r.lt.1.0D0) then
         print *, 'float gt lt double'
         print *, r
      endif

      if (r.gt.0.0E0.and.r.lt.1.0E0) then
         print *, 'float gt lt real'
         print *, r
      endif

! combines

      if (2*r.gt.0.and.2.*r.lt.1) then
         print *, 'float gt lt int int comb'
         print *, r
      endif

      if (2.0*r.gt.0.and.2.0*r.lt.1) then
         print *, 'float gt lt int float comb'
         print *, r
      endif

      if (2*r.gt.0.0.and.2*r.lt.1.0) then
         print *, 'float gt lt float int comb'
         print *, r
      endif

      if (2.0*r.gt.0.0.and.2.0*r.lt.1.0) then
         print *, 'float gt lt float float comb'
         print *, r
      endif

! float ge le

      if (r.ge.0.and.r.le.1) then
         print *, 'float ge le int'
         print *, r
      endif

      if (r.ge.0.0.and.r.le.1.0) then
         print *, 'float ge le real'
         print *, r
      endif

      if (r.ge.0.0D0.and.r.le.1.0D0) then
         print *, 'float ge le double'
         print *, r
      endif

      if (r.ge.0.0E0.and.r.le.1.0E0) then
         print *, 'float ge le float'
         print *, r
      endif

! double gt lt

      if (d.gt.0.and.d.lt.1) then
         print *, 'double gt lt int'
         print *, d
      endif

      if (d.gt.0.0.and.d.lt.1.0) then
         print *, 'double gt lt float'
         print *, d
      endif

      if (d.gt.0.0D0.and.d.lt.1.0D0) then
         print *, 'double gt lt double'
         print *, d
      endif

      if (d.gt.0.0E0.and.d.lt.1.0E0) then
         print *, 'double gt lt real'
         print *, d
      endif

! double ge le

      if (d.ge.0.and.d.le.1) then
         print *, 'double ge le int'
         print *, d
      endif

      if (d.ge.0.0.and.d.le.1.0) then
         print *, 'double ge le float'
         print *, d
      endif

      if (d.ge.0.0D0.and.d.le.1.0D0) then
         print *, 'double ge le double'
         print *, d
      endif

      if (d.ge.0.0E0.and.d.le.1.0E0) then
         print *, 'double ge le real'
         print *, d
      endif

      print *, i, r, d

! known - out
      
      i = 2
      r = 2.0
      d = 2.0D0

! int gt lt

      if (i.gt.0) then
         print *, 'int gt int'
         print *, i
      endif

      if (i.gt.0.and.i.lt.1) then
         print *, 'int gt lt int'
         print *, i
      endif

      if (i.gt.0.0.and.i.lt.1.0) then
         print *, 'int gt lt real'
         print *, i
      endif

      if (i.gt.0.0D0.and.i.lt.1.0D0) then
         print *, 'int gt lt double'
         print *, i
      endif

      if (i.gt.0.0E0.and.i.lt.1.0E0) then
         print *, 'int gt lt double'
         print *, i
      endif

! int ge le

      if (i.ge.0.and.i.le.1) then
         print *, 'int ge le int'
         print *, i
      endif

      if (i.ge.0.0.and.i.le.1.0) then
         print *, 'int ge le real'
         print *, i
      endif

      if (i.ge.0.0D0.and.i.le.1.0D0) then
         print *, 'int ge le double'
         print *, i
      endif

      if (i.ge.0.0E0.and.i.le.1.0E0) then
         print *, 'int ge le float'
         print *, i
      endif

! float gt lt

      if (r.gt.0.and.r.lt.1) then
         print *, 'float gt lt int'
         print *, r
      endif

      if (r.gt.0.0.and.r.lt.1.0) then
         print *, 'float gt lt float'
         print *, r
      endif

      if (r.gt.0.0D0.and.r.lt.1.0D0) then
         print *, 'float gt lt double'
         print *, r
      endif

      if (r.gt.0.0E0.and.r.lt.1.0E0) then
         print *, 'float gt lt real'
         print *, r
      endif

! combines

      if (2*r.gt.0.and.2.*r.lt.1) then
         print *, 'float gt lt int int comb'
         print *, r
      endif

      if (2.0*r.gt.0.and.2.0*r.lt.1) then
         print *, 'float gt lt int float comb'
         print *, r
      endif

      if (2*r.gt.0.0.and.2*r.lt.1.0) then
         print *, 'float gt lt float int comb'
         print *, r
      endif

      if (2.0*r.gt.0.0.and.2.0*r.lt.1.0) then
         print *, 'float gt lt float float comb'
         print *, r
      endif

! float ge le

      if (r.ge.0.and.r.le.1) then
         print *, 'float ge le int'
         print *, r
      endif

      if (r.ge.0.0.and.r.le.1.0) then
         print *, 'float ge le real'
         print *, r
      endif

      if (r.ge.0.0D0.and.r.le.1.0D0) then
         print *, 'float ge le double'
         print *, r
      endif

      if (r.ge.0.0E0.and.r.le.1.0E0) then
         print *, 'float ge le float'
         print *, r
      endif

! double gt lt

      if (d.gt.0.and.d.lt.1) then
         print *, 'double gt lt int'
         print *, d
      endif

      if (d.gt.0.0.and.d.lt.1.0) then
         print *, 'double gt lt float'
         print *, d
      endif

      if (d.gt.0.0D0.and.d.lt.1.0D0) then
         print *, 'double gt lt double'
         print *, d
      endif

      if (d.gt.0.0E0.and.d.lt.1.0E0) then
         print *, 'double gt lt real'
         print *, d
      endif

! double ge le

      if (d.ge.0.and.d.le.1) then
         print *, 'double ge le int'
         print *, d
      endif

      if (d.ge.0.0.and.d.le.1.0) then
         print *, 'double ge le float'
         print *, d
      endif

      if (d.ge.0.0D0.and.d.le.1.0D0) then
         print *, 'double ge le double'
         print *, d
      endif

      if (d.ge.0.0E0.and.d.le.1.0E0) then
         print *, 'double ge le real'
         print *, d
      endif

      print *, i, r, d

      end
