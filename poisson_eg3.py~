#fenics code for -(laplace)=f(x)
#here f(x)=2*(omega*pi)^2*u_e
#u_e= sin(omega*pi*x)*sin(omega*pi*y) 	

from dolfin import*
from mshr import*
import numpy

omega=1.0
#generate mesh and define function space
mesh=UnitSquareMesh(10,10)
V=FunctionSpace(mesh, "Lagrange", 1)

#define boundary condition u=0
def boundary(x,on_boundary):
	return on_boundary 
bc=DirichletBC(V,Constant(0.0),boundary)

u_exact=Expression("sin(%g*pi*x[0]) *sin(%g*pi*x[1])" %(omega, omega))

#define Variational Problem
w=TrialFunction(V)
v=TestFunction(V)
a=inner(nabla_grad(w),nabla_grad(v))*dx
f=2*(pow((omega*pi),2))*u_exact
L=f*v*dx

#compute solution
w=Function(V)
problem=LinearVariationalProblem(a,L,w,bc)
solver=LinearVariationalSolver(problem)
solver.solve

#plot solutions and mesh
plot(mesh, title="mesh over scaled domain")
plot()


