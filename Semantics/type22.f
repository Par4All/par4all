      program type22

C     Variable and constant lengths must be taken into account and substring applied

      character*3 s1
      character*5 s2

C     Should be truncated
      s1 = "Hello World!"

C     Should be (implictly) SPACE padded
      s2 = "Z"

      end
