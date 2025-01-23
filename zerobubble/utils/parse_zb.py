schedules = {
    4:
'''
FFFFFFFBFBWFBBWFBBWFBBWFWWWWBWBWBWBW
.FFFFFBFBFBFBWFBBWFBBWFBWFBWWBWWBWWWW
..FFFBFBFBFBFBFBWFBBWFBWFBWFBWWBWWWWWW
...FBFBFBFBFBFBFBFBWFBWFBWFBWFBWWWWWWWW
''',
    3:
'''
FFFFFFBWBWBFBFWBFWWBWBWBWBW
.FFFFBWBWBFBFWBFBFWBFBWWBWWW
..FBFBFBFBWFBFBWFBFBWFBWWWWWW
''',
    2:
'''
FFFFBWFBBWFWBWBWBW
.FBFBFBFBWFBWFBWWWW
'''
}

for n in range(2, 5):

    output_fn = f'schedule_{n}.txt'
    schedule = schedules[n]

    with open(output_fn, 'w') as f:
        for line in schedule.split("\n"):
            if len(line) == 0:
                continue
            count = {'F': 0, 'B': 0, 'W': 0}
            char_map = {'F': 'F', 'B': 'Bi', 'W': 'Bw'}
            for i, char in enumerate(line):
                if char == '.':
                    continue
                f.write(f"{char_map[char]}{count[char]}")
                count[char] += 1
                if i != len(line) - 1:
                    f.write(" ")
            f.write("\n")
