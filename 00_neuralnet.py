import numpy as np

def sig(x):
    a=1/(1+np.exp(-x))
    return a

def sig_der(x):
    return x*(1-x)

t_in=np.array([[0,0,1],
               [1,1,1],
               [1,0,1],
               [0,1,1]])
t_out=np.array([[0,1,0,1]]).T

print("input array")
print(t_in)
print("output array")
print(t_out)

np.random.seed(1)
weights=2*np.random.random((3,1))-1
print("before weights")
print(weights)

for i in range(20000):
    in_layer=t_in
    output=sig(np.dot(in_layer,weights))
    e=t_out-output
    adj=e*sig_der(output)
    weights+=np.dot(in_layer.T,adj)
print("after weights")
print(weights)
print("outputs")
for i in output:
    print(i,end="")
    if i[0]>0.50000:
        print("---[1]")
    else:
        print("---[0]")

        a=int(input(" "))