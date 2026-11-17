pA = 0.60
pB = 0.40

dA = 0.02
dB = 0.05

pD = (pA * dA) + (pB * dB)

pA_given_D = (pA * dA) / pD
pB_given_D = (pB * dB) / pD

print("Probability of defective product:", pD)
print("Probability from Machine A:", pA_given_D)
print("Probability from Machine B:", pB_given_D)
