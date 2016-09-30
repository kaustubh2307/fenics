# fenics code for notched square plate subjected to tensile load 

from dolfin import*
from mshr import*
import numpy

# define mesh and function space
mesh=UnitSquareMesh(100,100)
V=FunctionSpace(mesh,"Lagrange",1)

# define boundary conditions
class bottom(SubDomain):
	def inside(self,x,on_boundary):
		return x[1]==0 and on_boundary
class middle(SubDomain):
	def inside(self,x,on_boundary):
		return x[1]==0.5 and x[0]<=0.5 

bottom=bottom()
middle=middle()


u0=Constant(0.0)
bc1=DirichletBC(V,u0,bottom)

p0=Constant(1.0)
bc2=DirichletBC(V,p0,middle)

bc=[bc1,bc2]
#Define variational form
phi=TrialFunction(V)
v=TestFunction(V)
l0=0.1
a=l0*(inner(nabla_grad(phi),nabla_grad(v)))*dx +(1/l0)*(inner(phi,v)*dx)
f=Constant(0.0)
L=f*v*dx

# compute solutions
phi=Function(V)
solve(a==L,phi,bc)
plot(phi,axes=True)
interactive()



