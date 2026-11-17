prevalence = 0.01
sensitivity = 0.95
specificity = 0.90

false_positive = 1 - specificity

p_positive = (
    sensitivity * prevalence
    + false_positive * (1 - prevalence)
)

result = (sensitivity * prevalence) / p_positive

print("Probability of disease after positive test:", result)
print("Percentage:", result * 100)
