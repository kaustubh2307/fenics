# Fenics code for (-Laplace)=f(x,y) on x^2+y^2=R^2
# f(x,y)=4e^(-0.5((Rx-x0)/(sigma))^2-0.5((Ry-y0)/(sigma))^2)
# bc w=0

from dolfin import *
from mshr import *

# Set pressure function:
T = 10.0 # tension
A = 1.0  # amplitude of pressure
R = 0.3  # radius of domain
theta = 0.2
x0 = 0.6*R*cos(theta)
y0 = 0.6*R*sin(theta)
sigma = 0.025 

pressure = "4*exp(-0.5*(pow((%g*x[0] - %g)/%g, 2)) "\
		"- 0.5*(pow((%g*x[1] - %g)/%g, 2)))" % \
	(R, x0, sigma, R, y0, sigma)


#create mesh and function space
mesh = generate_mesh(Circle(dolfin.Point(0,0),2),32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition w=0
def boundary(x, on_boundary):
	return on_boundary
w=Constant(0.0)
bc = DirichletBC(V,w, boundary)

#Define variational problem
w= TrialFunction(V)
v= TestFunction(V)
f= Expression(pressure)
a= inner(nabla_grad(w), nabla_grad(v))*dx
L= v*f*dx

# Compute solution
w=Function(V)
problem = LinearVariationalProblem(a,L,w,bc)
solver=LinearVariationalSolver(problem)
solver.solve()

#energy
energy=0.5*pow(((A*R)/(8*pi*sigma)),2)*inner(nabla_grad(w),nabla_grad(w))*dx
E=assemble(energy)
print("Elastic_energy is", E)

# Plot solution and mesh
plot(mesh, title="Mesh over scaled domain")
plot(w, title="Scaled deflection")
f= interpolate(f, V)
plot(f, title="Scaled pressure")

# Find maximum real deflection
max_w = w.vector().array().max()
max_D = (A*max_w)/(8*pi*sigma*T)
print ("Maximum real deflection is", max_D)

# Verification for "flat" pressure (big sigma)
if sigma >= 50:
	w_exact = Expression("1 - x[0]*x[0] - x[1]*x[1]")
	w_e = interpolate(w_exact, V)
	w_e_array = w_e.vector().array()
	w_array = w.vector().array()
	diff_array = abs(w_e_array - w_array)
	print "Verification of the solution, max difference is %.4E" % \
		diff_array.max()
interactive()
