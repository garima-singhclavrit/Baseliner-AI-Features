

def func(*args):
    d,e,f = None,None,None
    l = [d,e,f]
    for i,j in args,l:
        args[i] = l[j]
    print(f"value of a is : {d}")
    print(f"value of b is : {e}")



a =1
b = 2
c = 3
list = [a,b,c]
func(*list)