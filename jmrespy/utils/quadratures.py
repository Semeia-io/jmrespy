"""
This file implements function wich returns all elements we need to compute gaussian quadratures
"""

####################### -------------------- Modules and libraries -------------------- #######################
import numpy as np





####################### -------------------- Gauss_Kronrod -------------------- #######################
"""
    Function which returns Gauss-Kronrod quadrature nodes and weights
Args:
    k (7 or 15): Number of quadrature points
"""
def Gauss_Kronrod(k):

    #Quadrature nodes
    nodes = np.array([
        -0.949107912342758524526189684047851,
        -0.741531185599394439863864773280788,
        -0.405845151377397166906606412076961,
        0,
        0.405845151377397166906606412076961,
        0.741531185599394439863864773280788,
        0.949107912342758524526189684047851,
        -0.991455371120812639206854697526329,
        -0.864864423359769072789712788640926,
        -0.586087235467691130294144838258730,
        -0.207784955007898467600689403773245,
        0.207784955007898467600689403773245,
        0.586087235467691130294144838258730,
        0.864864423359769072789712788640926,
        0.991455371120812639206854697526329
    ])

    if k == 7:

        #Quadrature weights for k = 7 nodes
        weights = np.array([
            0.129484966168869693270611432679082,
            0.279705391489276667901467771423780,
            0.381830050505118944950369775488975, 
            0.417959183673469387755102040816327,
            0.381830050505118944950369775488975,
            0.279705391489276667901467771423780,
            0.129484966168869693270611432679082
        ])

        #Selection of corresponding nodes
        nodes = nodes[0:6]

    
    #Quadrature weights for k = 15 nodes
    elif k == 15:
        weights = np.array([
            0.063092092629978553290700663189204,
            0.140653259715525918745189590510238,
            0.190350578064785409913256402421014,
            0.209482141084727828012999174891714,
            0.190350578064785409913256402421014,
            0.140653259715525918745189590510238,
            0.063092092629978553290700663189204,
            0.022935322010529224963732008058970,
            0.104790010322250183839876322541518,
            0.169004726639267902826583426598550,
            0.204432940075298892414161999234649,
            0.204432940075298892414161999234649,
            0.169004726639267902826583426598550,
            0.104790010322250183839876322541518,
            0.022935322010529224963732008058970
        ])

    return(weights, nodes)





####################### -------------------- Gauss_Hermite -------------------- #######################
"""
    Function which returns Gauss-Hermite quadrature nodes and weights, this function is exactly the same than gauher() from Dimitris Rizopoulos R package JM transcripted for python
Args:
    h (int): Number of quadrature points
"""
def Gauss_Hermite(h):

    m = int(np.floor((h + 1)/2))
    nodes = np.repeat(-1.0, h)
    weights = np.repeat(-1.0, h)

    for i in range(m):

        if i ==0:
            z = np.sqrt(2*h + 1) - 1.85575 * (2*h + 1)**(-0.16667)
        elif i==1:
            z = z - 1.14 * h**0.426 / z
        elif i==2:
            z = 1.86 * z - 0.86 * nodes[0]
        elif i==3:
            z = 1.91 * z - 0.91 * nodes[1]
        else:
            z = 2*z - nodes[i - 2]
        
        for its in range(10):
            p1 = 0.751125544464943
            p2 = 0

            for j in range(h):
                p3 = p2
                p2 = p1
                p1 = z * np.sqrt(2/(j+1)) * p2 - np.sqrt((j)/(j+1)) * p3 

            pp = np.sqrt(2*h) * p2
            z1 = z
            z = z1 - p1/pp
            if abs(z - z1) <= 3e-14:
                break

        nodes[i] = z
        nodes[h - 1 - i] = -z
        weights[i] = 2/(pp**2)
        weights[h - 1 - i] = weights[i]

    return(weights, nodes)


