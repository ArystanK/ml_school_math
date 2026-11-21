# Notes for Programming Exercises 

## p1
Linearity must hold for every possible pair of vectors u, v in R^n and every scalar a in R. A single counterexample is enough to disprove a universal statement, so a random test can reliably prove that a map is not linear. However, checking finitely many random inputs can never guarantee that the condition holds everywhere, since a map could fail linearity only on specific inputs that were not sampled.

## p2
When n doubles, the running time of matmul grows by roughly a factor of 8. The nested loops compute each of the n^2 entries of the result using n multiplications, which gives n^3 multiplications in total. Doubling n increases the number of operations to (2n)^3 = 8n^3.

## p5
Going from n-1 to n increases the running time of det_cofactor by roughly a factor of n, because expanding along the first row makes n recursive calls on (n-1)x(n-1) submatrices. Total work scales like n!. For n = 20 this would take millions of years, while det_rowreduce runs in O(n^3) time and finishes a 200x200 matrix in a fraction of a second.

## p8
As n increases, the Hilbert matrix becomes very ill-conditioned and the error max |x_i - 1| grows rapidly for all three methods. Among them, np.linalg.solve keeps the residual max |(Hx - b)_i| the smallest (close to machine epsilon), whereas forming the inverse explicitly with np.linalg.inv(H) @ b gives noticeably larger residuals. My own inverse(H) raises a singular matrix error starting at n = 8 (check your terminal for exact n), because some pivots during Gaussian elimination become smaller than tol = 1e-10 and rref treats those columns as non-pivot columns. This matches Remark 6.53: solving Ax = b directly is both faster and numerically more accurate than computing A^(-1) and multiplying by b.

## p9
First, (X^T X)^T = X^T (X^T)^T = X^T X, so X^T X is always symmetric.

Second, assume the columns of X are linearly independent and let X^T X w = 0. Multiplying on the left by w^T gives w^T X^T X w = (Xw)^T (Xw) = sum_i (Xw)_i^2 = 0. A sum of squares of real numbers is zero only when every term is zero, so Xw = 0. Since the columns of X are linearly independent, this forces w = 0. Therefore null(X^T X) = {0}, and by Corollary 7.11 the square matrix X^T X is invertible.

Third, fitting a parabola y = w0 + w1*x + w2*x^2 uses the monomial basis {1, x, x^2} of R[x]_{<=2}. Finding (w0, w1, w2) corresponds to finding the coordinate vector of the best-fitting polynomial in R[x]_{<=2} (which is isomorphic to R^3), and the problem remains linear in the coefficients w.

## p10
In power iteration, the components along non-dominant eigenvectors shrink by (|lambda_i / lambda_1|)^k at step k, so convergence speed is controlled by |lambda_2 / lambda_1|. A smaller ratio gives fast convergence; a ratio close to 1 gives slow convergence.

For the 90-degree rotation, there are no real eigenvalues (the complex eigenvalues +-i both have absolute value 1), so the vector keeps rotating every step and never converges. For diag(1, -1), both eigenvalues have the same absolute value |1| = |-1| = 1, so the second coordinate flips sign on every iteration and the sequence oscillates between two vectors.

For the 4-page web graph, power iteration converges to the stationary eigenvector of G with eigenvalue 1. The pages ranked from most to least important are 3, 1, 2, 4.

## p11
Binet's formula gives a wrong answer for the first time at n = 71. Standard 64-bit floats only store 53 bits of significand (about 15 to 17 decimal digits), so once F_n exceeds that precision, rounding errors in phi^n / sqrt(5) become larger than 0.5 and rounding fails to recover the exact integer.

For the shear matrix [[1, 1], [0, 1]], the eigenvalue 1 has algebraic multiplicity 2 but geometric multiplicity 1 (only e_1 is an eigenvector), so the matrix is not diagonalizable. Still, np.linalg.eig returns two columns that are nearly collinear (one is [1, 0] and the second is [-1, ~1e-16] due to floating-point rounding).

## p12
If the columns of C are the coefficient vectors of the Lagrange polynomials L_0, ..., L_d in the monomial basis, then entry (k, j) of the product V @ C equals L_j(t_k). The defining condition L_j(t_k) = delta_{jk} is therefore V @ C = I, which means C = V^(-1). For nodes -1, 0, 1, the columns of V^(-1) are [0, -0.5, 0.5]^T, [1, 0, -1]^T, and [0, 0.5, 0.5]^T, which correspond to (-x + x^2)/2, 1 - x^2, and (x + x^2)/2, matching Example 8.9.

For sin(3t) on [-1, 1] with 9 nodes, the interpolant matches the function closely. For 1 / (1 + t^2) on [-5, 5] with 21 equally spaced nodes, the polynomial matches all nodes but develops huge oscillations near the endpoints -5 and 5 (Runge's phenomenon).

## p13
On z = (1001, 1002, 1003), softmax_naive overflows because exp(1001) exceeds the maximum float64 value (~1.8e308, around exp(709.78)), resulting in inf / inf = nan. Subtracting max(z) shifts z within the same coset z + span{1}, which cancels out as a common factor exp(-max(z)) in numerator and denominator, while ensuring all shifted entries are <= 0 so exp(z_i - max(z)) stays in (0, 1] and never overflows.

Centering by subtracting the mean also stays in the same coset and gives the same mathematical result, but in practice it can still overflow. If a vector has one large entry and many small entries (for example [0, 0, 0, 0, 0, 1000]), the mean stays small, so the largest entry after centering is still above 710 and exp overflows. Subtracting the maximum guarantees the largest entry is always exactly 0.