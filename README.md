# Introduction
This Python package is a proof-of-concept implementation of a binary search strong rank-revealing QR decomposition (`SRRQR_bin`).
It is an improved version of the strong rank-revealing QR decomposition (`SRRQR_l2r`), which is described as Algorithm 4 in:

```
 Ming Gu, Stanley C. Eisenstat,
"EFFICIENT ALGORITHMS FOR COMPUTING A STRONG RANK-REVEALING QR FACTORIZATION"
 https://math.berkeley.edu/~mgu/MA273/Strong_RRQR.pdf
```

`SRRQR_l2r` tries out every rank k from left (k=0) to right (k=min(m,n)) in order to find the correct rank of the matrix.
As the name already indicates, the binary search version `SRRQR_bin` uses binary search to find the actual rank. This greatly
reduces the number of "strong" column swaps required for larger matrices. While the threshold `f` cannot be constant if `SRRQR_l2r`
is to achieve `O(m^2*n)` performance, `SRRQR_bin` _should_ work with constant values for `f`.
