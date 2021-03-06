# fenics code for -(laplace)=f with dirichlet and neumann bc on unit square
# f=-6 ,u=u0=1+x.^2+2y.^2 on x=1 and x=0 ,neumann bc g=-4 on y=1 and g=0 on y=0

from dolfin import* 
import numpy 

#define mesh and function space
mesh=UnitSquareMesh(10,10)
V=FunctionSpace(mesh,'Lagrange',1)

# define Boundary Conditions
tol = 1E-14
u0=Expression('1+x[0]*x[0]+2*x[1]*x[1]')
def Dirichlet_boundary0(x,on_boundary):
	return on_boundary and abs(x[0])<tol

def Dirichlet_boundary1(x,on_boundary):
	return on_boundary and abs(x[1]-1)<tol

bc0=DirichletBC(V,u0,Dirichlet_boundary0)
bc1=DirichletBC(V,u0,Dirichlet_boundary1)
bc=[bc0,bc1]

#Define varaiational problem
u=TrialFunction(V)
v=TestFunction(V)
a=inner(nabla_grad(u),nabla_grad(v))*dx
f=Constant(-6.0)
g=Expression('4*x[1]')
L=f*v*dx -g*v*ds

#Compute solutions
u=Function(V)
solve(a==L,u,bc)

#plot
plot(u,axes=True,rescale=True) 
plot(mesh)
interactive()
