      program declarations
      dimension a(10)
      integer a
      common /a/ a
      dimension b(10)
      integer b
      save b
      dimension c(10)
      integer c
      dimension d(10)
      integer d
      data d / 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 /
      print *, 'declarations'
      end
