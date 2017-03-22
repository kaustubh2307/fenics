# function for notched square plate subjected to tensile load 

from dolfin import*
from mshr import*
import numpy


# define mesh and function space
def initial_notch(mesh,l0):
	V1=FunctionSpace(mesh,"Lagrange",1)

	# define boundary conditions
	class Middle(SubDomain):
		def inside(self,x,on_boundary):
			return x[1]==0.0 and x[0]<=0.0
	middle=Middle()

	p0=Constant(1.0)
	bc2=DirichletBC(V1,p0,middle)

	bc_i=[bc2]
	
	#Define variational form
	phi1=TrialFunction(V1)
	v1=TestFunction(V1)
	a1=l0*(inner(nabla_grad(phi1),nabla_grad(v1)))*dx +(1/l0)*(inner(phi1,v1)*dx)
	f1=Constant(0.0)
	L1=f1*v1*dx
	
	# compute solutions
	phi1=Function(V1)
	solve(a1==L1,phi1,bc_i)
	plot(phi1,axes=True)
	interactive()
	return(phi1)
	


