import numpy as np 


def from_n_to_kF(n):
    
    k_F = np.sqrt((n) * 2 * np.pi)
    
    a = 3.728 *1e-9
    
    return k_F*a


print(from_n_to_kF(1e11))
print(from_n_to_kF(2e12))


