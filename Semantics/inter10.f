C     Since i is not used by foo, the precondition i==5 will not be
C     displayed as it is rightly assumed to be useless. Uncomment the
C     "print *, i" to see the precondition.

C     It is not clear if this feature is helpful, or too surprising for
C     a standard PIPS user who will assume an internal error in PIPS
C     when not seeing information about i.
      subroutine foo(i)
C      print *, i
      end

      integer function inter10(i)
      inter10 =  i+1
      end

      program main
      integer i
      data i/4/
      i = inter10(i)
      call foo(i)
      end

