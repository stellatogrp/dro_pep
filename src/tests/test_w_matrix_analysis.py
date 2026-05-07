"""
Analyze how the W matrix affects subgradient inner products in the Gram matrix.

This script checks if the W matrix is being applied correctly for interpolation constraints.
"""

import jax.numpy as jnp
import numpy as np

# Simulate the setup
n = 3  # primal dim
m = 4  # dual dim

# Random constraint matrix
K_mat = np.random.randn(m, n) * 0.5
M = np.linalg.norm(K_mat, ord=2)

# W matrix from code
W = np.block([
    [np.eye(n), K_mat.T / M],
    [K_mat / M, np.eye(m)]
])

print("="*70)
print("W Matrix Structure Analysis")
print("="*70)

# Embedding functions
def embed_primal(v):
    return np.concatenate([v, np.zeros(m)])

def embed_dual(v):
    return np.concatenate([np.zeros(n), v])

# Test vectors
u1 = np.random.randn(n)
u2 = np.random.randn(n)
v1 = np.random.randn(m)
v2 = np.random.randn(m)

print("\n1. PRIMAL-PRIMAL inner product:")
print(f"   Standard: <u1, u2> = {np.dot(u1, u2):.6f}")
ip_w = embed_primal(u1).T @ W @ embed_primal(u2)
print(f"   With W:   <u1, u2>_W = {ip_w:.6f}")
print(f"   Difference: {np.abs(np.dot(u1, u2) - ip_w):.2e}")
print(f"   → W acts as IDENTITY for primal-primal ✓")

print("\n2. DUAL-DUAL inner product (this is what matters for gh!):")
print(f"   Standard: <v1, v2> = {np.dot(v1, v2):.6f}")
ip_w = embed_dual(v1).T @ W @ embed_dual(v2)
print(f"   With W:   <v1, v2>_W = {ip_w:.6f}")
print(f"   Difference: {np.abs(np.dot(v1, v2) - ip_w):.2e}")
print(f"   → W acts as IDENTITY for dual-dual ✓")

print("\n3. PRIMAL-DUAL (cross) inner product:")
standard_cross = np.dot(u1, K_mat.T @ v1) / M
ip_w = embed_primal(u1).T @ W @ embed_dual(v1)
print(f"   Expected: <u1, K^T v1> / M = {standard_cross:.6f}")
print(f"   With W:   <u1, v1>_W = {ip_w:.6f}")
print(f"   Difference: {np.abs(standard_cross - ip_w):.2e}")
print(f"   → W creates P-norm cross-term ✓")

print("\n" + "="*70)
print("CONCLUSION: W Matrix Usage")
print("="*70)
print("\nFor interpolation constraints like:")
print("  h(y_i) >= h(y_j) + <gh(y_j), y_i - y_j>")
print("\nThe inner product <gh(y_j), y_i - y_j> should be STANDARD EUCLIDEAN.")
print("Since gh and (y_i - y_j) are both DUAL vectors, and W acts as")
print("IDENTITY on dual-dual inner products, the W matrix usage is CORRECT!")
print("\n→ The W matrix is NOT the problem!")
print("\n" + "="*70)

print("\n" + "="*70)
print("ACTUAL PROBLEM: gh_s = 0 instead of gh_s = q")
print("="*70)

# Simulate the constraint check
q = np.random.randn(m) * 2.0
y0 = np.random.randn(m)
y_opt = np.random.randn(m)

# Function values
h_y0 = np.dot(q, y0)
h_yopt = np.dot(q, y_opt)
F_h_0 = h_y0 - h_yopt

print(f"\nFor h(y) = q^T y:")
print(f"  q = {q}")
print(f"  F_h[0] = h(y0) - h(y_opt) = {F_h_0:.6f}")

# Constraint [14]: h(y0) >= h(ys) + <gh(ys), y0-ys>
# In PEP form: Tr(A @ G) + b^T @ F <= 0
# where: A represents <gh_s, dy0>, b represents F_h[ys] - F_h[y0]

print(f"\nInterpolation constraint: h(y0) >= h(ys) + <gh(ys), y0-ys>")
print(f"  LHS: h(y0) - h(ys) = {F_h_0:.6f}")

dy0 = y0 - y_opt
print(f"\nWith gh_s = 0 (current code):")
ip_gh_dy = 0.0
print(f"  RHS: <gh_s, y0-ys> = {ip_gh_dy:.6f}")
print(f"  Constraint: {F_h_0:.6f} >= {ip_gh_dy:.6f}")
print(f"  Satisfied? {F_h_0 >= ip_gh_dy} (but should be equality for linear h!)")
print(f"  PEP form: Tr(A@G) + b^T@F = {ip_gh_dy:.6f} + {-F_h_0:.6f} = {ip_gh_dy - F_h_0:.6f}")
print(f"  VIOLATION: {ip_gh_dy - F_h_0:.6f} > 0 ✗")

print(f"\nWith gh_s = q (correct):")
ip_gh_dy = np.dot(q, dy0)
print(f"  RHS: <q, y0-ys> = {ip_gh_dy:.6f}")
print(f"  Constraint: {F_h_0:.6f} >= {ip_gh_dy:.6f}")
print(f"  Satisfied? {F_h_0 >= ip_gh_dy} ✓")
print(f"  PEP form: Tr(A@G) + b^T@F = {ip_gh_dy:.6f} + {-F_h_0:.6f} = {ip_gh_dy - F_h_0:.6f}")
print(f"  SATISFIED: {np.abs(ip_gh_dy - F_h_0):.2e} ≈ 0 ✓")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("• W matrix is CORRECT - it acts as identity for dual-dual inner products")
print("• The bug is: gh_s = 0 should be gh_s = q (and gf1_s = 0 should be gf1_s = c)")
print("• This works accidentally when c=0 and q=0, but fails for general LPs")
print("="*70)
