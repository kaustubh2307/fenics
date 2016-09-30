from  dolfin  import *
from mshr import *
import numpy
import math

# generate mesh and function space
Length=10
height=4
g1=Rectangle(Point(0,0),Point(Length,height))
g2=Circle(Point(5,2),1)
domain=g1-g2
mesh=generate_mesh(domain,20)
V = VectorFunctionSpace(mesh ,"Lagrange",1)

# define material property
E = 200e9     
nu = 0.3
lmbda = E*nu /((1.0 + nu)*(1.0-2*nu)) 
mu = E / (2 * (1 + nu))
def  epsilon(v):
	D=nabla_grad(v)
	return  0.5 * (D+D.T)
def  sigma(v):
	return 2 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity (2)

#define boundary conditions and mark boundaries
tol=1E-14
class left(SubDomain):
	def inside(self,x,on_boundary):
		return abs(x[0])<=tol and on_boundary

class right(SubDomain):
	def inside(self,x,on_boundary):
		return abs(x[0]-10)<=tol and on_boundary
left=left()
right=right()

boundaries=FacetFunction("size_t",mesh)
boundaries.set_all(0)
left.mark(boundaries,1)
right.mark(boundaries,2)


u0=Constant(("0.0","0.0"))
bc1=DirichletBC(V,u0,boundaries,1)

ds=Measure("ds")[boundaries]

bc=[bc1]

# define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f=Constant(("0.0","0.0"))   # body force
g=Expression(("100e3","0.0"))   # traction 
a = (inner(nabla_grad(v), sigma(u)))*dx
L = inner(v,f)*dx+inner(g,v)*ds(2)

# compute solutions
u = Function(V)
solve(a==L,u,bc)

u_values=u.vector()
u_array=u_values.array()
print(u_array)

# plot 
plot(u,mode="displacement",axes=True)
plot(mesh)
interactive()








