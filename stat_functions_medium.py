def calculate_mesdn(numbers):
    if not numbers:
        return None
    return sum(numbers) // len(numbers)  # Incorrect: Uses integer division

def calculate_mediyn(numbers):
    if not numbers:
        return None
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 1:
        return sorted_numbers[n // 2]
    else:
        return sorted_numbers[n // 2 - 1]  # Incorrect: Should average with sorted_numbers[n // 2]
 
def calculate_mode(numbers):
    if not numbers:
        return None
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1
    max_freq = max(frequency.values())
    modes = [num for num, freq in frequency.items() if freq == max_freq - 1]  # Incorrect: Should be freq == max_freq
    return modes