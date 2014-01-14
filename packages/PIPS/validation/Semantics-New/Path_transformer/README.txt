Validation of path transformer computation

Different dimensions are used to classify the test cases:

 1. constant vs symbolic loop bounds

 2. external loop(s) or not

 3. one or many loops involved at the same level

 4. computation of the transformers in context or out of context

 5. relative text position of the statements Sb, Se and S, where S is a loop
   and where all statements are a the same level, i.e. in the same
   sequence.

To debug cases with no external loops, we consider 6 different cases
for one sequence:

 if Sb < S then
    if Se > S
    else Se in S
 else Sb in S
    if Se > S
    else Se in S // We no longer have one sequence for S, Se and Sb
       if Sb < Se
       else if Sb == Se
       else Sb > Se // Possible because of loop S

where S1 < S2 means that S1 is textually before S2.

Check for behaviors with no external loops and symbolic bounds (criteria 1 and 2 are fixed)

sb_equal_se.c
sb_equal_se_loop.c
sb_loop_in_ctxt.c
sb_loop_out_ctxt.c
sb_loop_se_loop.c
sb_se_loop.c
se_loop.c
sequence.c
sequence_sb_loop_se.c

Check for behavior in tests with no external loop:

se_test_false.c
se_test_true.c
