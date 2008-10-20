      program main
      common /w/ w(1)
      c = .true.
      if (c) then
         x = w(1)
         call p
      else
         call q
      endif
      end
      subroutine p
      common /v/ v(1)
      t =.true.
      if (t) then
         v(1) = 4
      endif
      call r
      end
      subroutine q
      common /w/ w(1)
      print *,w(1)
      call r
      call s
      end
      subroutine r
      common /v/ v(1)
      t = .true.
      if (t) then
c         l = v(1)
         call foo
      endif
      end
      subroutine s
      common /w/ w(1)
      common /v/ v(1)
      t = .true.
      if (t) then
         a = w(1)
      else
         b = v(1)
      endif
      end
      subroutine foo
      common /w/ w(1)
      common /v/ v(1)
      t = .true.
      if (t) then
         a = w(1)
      else
         b = v(1)
      endif
      end

