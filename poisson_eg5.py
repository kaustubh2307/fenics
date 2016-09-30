# fenics problem: -(laplace)=f with multiple dirichlet and neumann BC
# f=-6; uL=1+2y.^2 on x=0;uR=2+2y.^2 on x=1;g=-4y 

from dolfin import*
import numpy 

#define mesh and function space
mesh=UnitSquareMesh(10,10)
V=FunctionSpace(mesh,'Lagrange',1)

# boundary conditions
tol=1E-14
uL=Expression('1+2*x[1]')
def Left_Dirichlet_onBoundary(x,on_boundary):
	return on_boundary and abs(x[0])<tol
bcL=DirichletBC(V,uL,Left_Dirichlet_onBoundary)

uR=Expression('2+2*x[1]')
def Right_Dirichlet_onBoundary(x,on_boundary):
	return on_boundary and abs(x[0]-1)<tol
bcR=DirichletBC(V,uR,Right_Dirichlet_onBoundary)

bc=[bcL,bcR]

# Define variational problem
u=TrialFunction(V)
v=TestFunction(V)
a=inner(nabla_grad(u),nabla_grad(v))*dx
f=Constant(-6.0)
g=Expression('-4*x[1]')
L=f*v*dx-g*v*ds

# compute solution
u=Function(V)
solve(a==L,u,bc)

#plot solutions
plot(mesh)
plot(u,axes=True)
interactive()





