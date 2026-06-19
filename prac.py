def generate_num(n):
    i = 0
    try:
        while i < n:
            yield i*i
            i+=1
    except:
        print(f'The code exit with')

gen = generate_num(5)

print(type(gen))
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))
