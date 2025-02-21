def calculate_mean(numbers):
    if not numbers:
        return None
    return sum(numbers)  # Incorrect: Should divide by len(numbers)

def calculate_median(numbers):
    if not numbers:
        return None
    n = len(numbers)
    return numbers[n // 2]  # Incorrect: Should sort the list first

# calculate_mode is missing