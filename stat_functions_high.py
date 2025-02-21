def calculate_mean(numbers):
    if not numbers:
        return None
    return sum(numbers) / len(numbers)  # Correct

def calculate_median(numbers):
    if not numbers:
        return None
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 1:
        return sorted_numbers[n // 2]
    else:
        return (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2  # Correct

def calculate_mode(numbers):
    if not numbers:
        return None
    from collections import Counter
    counter = Counter(numbers)
    max_freq = max(counter.values())
    modes = [num for num, freq in counter.items() if freq == max_freq]  # Correct, using Counter for efficiency
    return modes