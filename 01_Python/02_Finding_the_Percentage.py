# HackerRank: Finding the Percentage
# Calculate a student's average marks and print it to 2 decimal places.

if __name__ == '__main__':
    n = int(input())
    marks = {}

    for _ in range(n):
        name, *scores = input().split()
        marks[name] = list(map(float, scores))

    query_name = input()

    average = sum(marks[query_name]) / len(marks[query_name])
    print(f"{average:.2f}")
