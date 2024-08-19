import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

"""
Numerical solution for Navier-Stokes Equation

Mesh:
         |                 |
    -----------v(i,j)------------
         |                 |
     u(i-1,j)  p(i,j)    u(i,j)
         |                 |
    ----------v(i,j-1)-----------
         |                 |

- N - mesh size (square NxN)
- eps - discrepancy (error of solution)
- nu = 1 / Re , Re - Reynolds number [0, +inf]
- alpha - relaxation coefficient 

Used SIMPLE (Semi-Implicit Method for Pressure Linked Equations, Patankar & Spalding, 1972) method. : Explicit Finite Volume Method
          +
Relaxation method: 
    instead: p = p* + p'
    
    do: u* <-- alpha_u * u* + (1-alpha_u)*u
        p = p* + alpha_p * p'
        
    for example: alpha_u = 0.5, alpha_p = 0.8 
    
https://cfdisraelblog.com/2021/11/08/simple-algorithm-way-to-solve-incompressible-nv-stokes-equation/   
"""

class Navier:
    def __init__(self, N, eps, nu, alpha_p, alpha_u):

        self.A = np.zeros((N * N, N * N))
        self.eps = eps
        self.h = 1 / N
        self.N = N
        self.alpha_p = alpha_p
        self.alpha_u = alpha_u
        self.nu = nu
        
        #laplas_pressure coef
        for i in range(N):
            for j in range(N):
                ij = i * N + j
                counter = 0
                if ij > N - 1:
                    self.A[ij, ij - N] = -1
                    counter += 1
                if ij % N != 0:
                    self.A[ij, ij - 1] = -1
                    counter += 1
                if (ij + 1) % N != 0:
                    self.A[ij, ij + 1] = -1
                    counter += 1
                if ij < N * (N - 1):
                    self.A[ij, ij + N] = -1
                    counter += 1
                self.A[ij, ij] = counter
        self.u_prev = np.zeros((N, N + 1))
        self.v_prev = np.zeros((N + 1, N))
        self.p_prev = np.zeros(N * N)
        self.b = np.zeros(N * N)

    def solve_P(self, b):
        sol = np.linalg.solve(self.A, b)
        # print(sol)
        return sol
    

    def div(self, u, v, dt):
        b = np.zeros(self.N * self.N)
        for i in range(self.N):
            for j in range(self.N):
                b[i * self.N + j] = -self.h * (u[i, j + 1] - u[i, j] + v[i + 1, j] - v[i, j])

        return b
        
    def solve_UV(self, p, dt):
        '''
         du/dt = (u*grad)*u - nu (laplace(u)) + grad(p)

        TWO STEPS:
        du/dt = u*(du/dx)+v*(du/dy) - nu*(d2u/dx2+d2u/dy2) + px
        dv/dt = u*(dv/dx)+v*(dv/dy) - nu*(d2v/dx2+d2v/dy2) + px

        -p- pressure on previous iteration
        
        '''
        N = self.N
        u = np.zeros((N, N + 1))
        v = np.zeros((N + 1, N))

        for i in range(self.N):  # i = [0,N)
            for j in range(1, self.N):  #

                uw = self.u_prev[i, j - 1]
                up = self.u_prev[i, j]
                ue = self.u_prev[i, j + 1]
                 # upper boundary
                if i == 0: 
                    un = 2 - up
                else:
                    un = self.u_prev[i - 1, j]
                # lower boundary    
                if i == (self.N - 1):  
                    us = -up
                else:

                    us = self.u_prev[i + 1, j]
                vnw = self.v_prev[i, j - 1]
                vne = self.v_prev[i, j]
                vsw = self.v_prev[i + 1, j - 1]
                vse = self.v_prev[i + 1, j]
                # print(f'i: {i}, j: {j}')
                pe = p[i * N + j]
                pw = p[i * N + j - 1]

                gradU = 0.25 / self.h * (
                        (up + ue) * (up + ue) - (uw + up) * (uw + up) - (vnw + vne) * (un + up) + (
                        vsw + vse) * (us + up)
                )

                u[i, j] = up - dt * (
                        gradU + self.nu / (self.h ** 2) * (4 * up - uw - ue - us - un) + (pe - pw) / self.h)

        # --------------------------------------------------------------------------------

        for i in range(1, N):
            for j in range(N):
                vp = self.v_prev[i, j]
                vn = self.v_prev[i - 1, j]
                vs = self.v_prev[i + 1, j]
                
                # right boundary
                if j == N - 1: 
                    ve = -vp
                else:
                    ve = self.v_prev[i, j + 1]
                    
                 # left boundary    
                if j == 0: 
                    vw = -vp
                else:
                    vw = self.v_prev[i, j - 1]
                une = self.u_prev[i - 1, j + 1]
                unw = self.u_prev[i - 1, j]
                use = self.u_prev[i, j + 1]
                usw = self.u_prev[i, j]
                pn = p[(i - 1) * N + j]
                ps = p[i * N + j]
                
                gradV = 0.25 / self.h * (
                        (une + use) * (ve + vp) - (unw + usw) * (vp + vw) - ((vn + vp)) ** 2 + (
                    (vs + vp)) ** 2
                )
                v[i, j] = vp - dt * (gradV + self.nu / self.h ** 2 * (4 * vp - vw - ve - vs - vn) + (ps - pn) / self.h)

        return u, v


    def solver(self, dt, stop=0.01):
    
        N = self.N
        eps = self.eps
        alpha_p = self.alpha_p
        alpha_u = self.alpha_u
        t = 0
        
        
        check = True
        self.u_prev = np.zeros((N, N + 1))
        self.v_prev = np.zeros((N + 1, N))
        self.p_prev = np.zeros(N * N)


        while check and t < 30: 
            b_check = False
            # print(t)
            norm = eps + 1
            p = self.p_prev
            # while not b_check:
            it = -1
            while norm > eps:
                it += 1
                # b_check = True
                u, v = self.solve_UV(p, dt)
                b = self.div(u, v, dt)
                print('t: ', t, ' iter: ', it, ' norm: ', np.linalg.norm(b), ' norm_p', np.mean(p))
                # print(p)
                norm = np.linalg.norm(b)
                
                #Regularization for p'
                if norm > self.eps:
                    p_correction = self.solve_P(b / dt)
                    p += self.alpha_p * p_correction * self.h
                else:
                    b_check = True

            if np.linalg.norm(u - self.u_prev) < stop and np.linalg.norm(v - self.v_prev) < stop and t > 1:
                check = False
            # Regularization for u
            self.u_prev = self.alpha_u * u + (1 - self.alpha_u) * self.u_prev 
            self.v_prev = self.alpha_u * v + (1 - self.alpha_u) * self.v_prev
            self.p_prev = p

            t += dt
        clear_output()
        self.plot_solution(self.u_prev, self.v_prev, self.p_prev, dt)  # break
        # print('t: ', t)
        # t+=

    def plot_solution(self, u, v, p, dt, streamplot=True):

        u = (u[:, :-1] + u[:, 1:]) / 2
        v = (v[1:, :] + v[:-1, :]) / 2
        N = self.N

        print(u.shape)
        u = u[::-1, ::]
        v = -v[::-1, ::]
        p = p.reshape((N, N))[::-1, ::]
        x = np.arange(self.h / 2, 1, self.h)
        y = np.arange(self.h / 2, 1, self.h)
        grid_x, grid_y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(10, 10))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.streamplot(grid_x, grid_y, u, v, color='black')
        plt.contourf(grid_x, grid_y, p.reshape((self.N, self.N)))
        plt.title(f"N = {self.N}, Eps = {self.eps}, Nu = {self.nu}, dt = {dt}", fontsize=20)
        plt.savefig('navie-stoks_nu_' + str(self.nu) + '.png')
        plt.show()



A = Navier(N=25, eps=1e-3, nu=1e-3, alpha_p = 0.8, alpha_u = 0.5)
A.solver( 1, stop=0.01)
# print('u:',A.u_prev)
# print('v: ',A.v_prev)
# print(A.p_prev)
# print(A.A)


