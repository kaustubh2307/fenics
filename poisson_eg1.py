#Fenics code for -(laplace)=f(x,y) equation with dirichlet BC
#u_e=u0=1+x.^2+2y.^2
#f(x,y)=-6

from dolfin import *
import numpy

# Create mesh and define function space
mesh=UnitSquareMesh(1,1)
V=FunctionSpace(mesh,'Lagrange',1)

# Define Boundry Conditions
u0=Expression('1+x[0]*x[0]+2*x[1]*x[1]')

def u0_onboundary(x,on_boundary):
	return on_boundary

bc= DirichletBC(V,u0,u0_onboundary)

# Define variational problem
u=TrialFunction(V)
v=TestFunction(V)
f=Constant(-6.0)
a=inner(nabla_grad(u),nabla_grad(v))*dx
L=f*v*dx

# compute solution
u=Function(V)
problem=LinearVariationalProblem(a,L,u,bc)
solver=LinearVariationalSolver(problem)
solver.solve()

energy=0.5*inner (nabla_grad(u), nabla_grad(u))*dx
E=assemble(energy)

# u values and mapping of vertex and dof
 	#nodal values 
u_nodal_values=u.vector()
u_array=u_nodal_values.array()
print(u_array)

#coordinates of vertics
coor=mesh.coordinates()
print(coor)

#u(i) corespong to vertices 
u_at_vertices=u.compute_vertex_values()
for i,x in enumerate(coor):
	print('u[%8g,%8g]=%g' %
		(coor[i][0],coor[i][1],u_at_vertices[i]))

# Error
u_e=interpolate(u0,V)
u_e_array=u_e.vector().array()
print(u_e_array)
print('MAX error:',numpy.abs(u_e_array-u_array).max())

# point evalution 
center = (0.5,0.5)
u_value=u(center)
u0_value=u0(center)
print("numerical value at center:",u_value)
print("exact value at center:",u0_value)

#plot solution and mesh
plot(u)
plot(mesh)
# Dump sol to file in VTK format
file = File("poisson.pvd")
file << u

# Hold plot
interactive()
