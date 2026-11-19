# run3 mirror: draw states through lm_head, L12

mass_* = mean softmax probability on the concept's keyword tokens, x1e3.
argmax_kw_* = keyword regex hits over the 8x576 argmax tokens.

## baseline

top-30 argmax tokens: '<end_of_image>'x4608

argmax sequence, prompt 0, first 96 positions:

`'<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## ggb_r0.3

top-30 argmax tokens: '<end_of_image>'x4608

most over-represented vs baseline (token, count, baseline count): '<end_of_image>' 4608 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## ggb_r0.45

top-30 argmax tokens: '<end_of_image>'x4608

most over-represented vs baseline (token, count, baseline count): '<end_of_image>' 4608 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## ggb_po_r1.0

top-30 argmax tokens: '<end_of_image>'x4602, '\n'x6

most over-represented vs baseline (token, count, baseline count): '\n' 6 vs 0, '<end_of_image>' 4602 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'\n' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## ggb_po_r2.0

top-30 argmax tokens: '<end_of_image>'x4600, '\n'x6, ' Discover'x1, ' Lights'x1

most over-represented vs baseline (token, count, baseline count): '\n' 6 vs 0, ' Discover' 1 vs 0, ' Lights' 1 vs 0, '<end_of_image>' 4600 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'\n' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## coffee_r0.3

top-30 argmax tokens: '<end_of_image>'x4608

most over-represented vs baseline (token, count, baseline count): '<end_of_image>' 4608 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## coffee_r0.45

top-30 argmax tokens: '<end_of_image>'x4608

most over-represented vs baseline (token, count, baseline count): '<end_of_image>' 4608 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## circle_r0.45

top-30 argmax tokens: '<end_of_image>'x4608

most over-represented vs baseline (token, count, baseline count): '<end_of_image>' 4608 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## random_r0.45

top-30 argmax tokens: '<end_of_image>'x4608

most over-represented vs baseline (token, count, baseline count): '<end_of_image>' 4608 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

## random_s2_r0.45

top-30 argmax tokens: '<end_of_image>'x4608

most over-represented vs baseline (token, count, baseline count): '<end_of_image>' 4608 vs 4608

argmax sequence, prompt 0, first 96 positions:

`'<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>' '<end_of_image>'`

