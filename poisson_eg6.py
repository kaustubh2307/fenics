# fenics code for -(p(x,y)*laplace)=f(x,y) with variable coefficient poissin problem
# p(x,y)=x+y and f(x,y)=-8x-10y
# u=u0=1+x.^2+2y.^2 on boundary;

from dolfin import*
from mshr import*
import numpy

# define mesh and Function Space
mesh=UnitSquareMesh(10,10)
V=FunctionSpace(mesh,'Lagrange',1)

# define boundary Conditions
u0=Expression('1+x[0]*x[0]+2*x[1]*x[1]')
def boundary(x,on_boundary):
	return on_boundary
bc=DirichletBC(V,u0,boundary)

# define Variational problem
u=TrialFunction(V)
v=TestFunction(V)
p=Expression('x[0]+x[1]')
a=p*inner(nabla_grad(u),nabla_grad(v))*dx
f=Expression('-8*x[0]-10*x[1]')
L=f*v*dx

# compute solution 
u=Function(V)
solve(a==L,u,bc)

# plot
plot(mesh)
plot(u,axes=True)
interactive()

# gradient or flux
flux=project(-p*nabla_grad(u),VectorFunctionSpace(mesh,'Lagrange',1))
plot(flux,axes=True,title="flux")
flux_x, flux_y = flux.split(deepcopy=True)                                  # extract components
plot(flux_x, title="x-component of flux (-p*grad(u))")
plot(flux_y, title="y-component of flux (-p*grad(u))")
interactive()






