#-----------------------Fenics code for modified stress equilbrium equation ---------------------#
from  dolfin  import *
from mshr import *
import numpy as np
from notch import initial_notch
import matplotlib.pyplot as plt
import csv
#import xlsxwriter

# ---Define mesh and function space--------#
nx = 1
ny = 1
num = 1
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, 2, 2)
editor.init_vertices(10)
editor.init_cells(8)
editor.add_vertex(0, 0, 0.0)
editor.add_vertex(1, 0.5, 0.0)
editor.add_vertex(2, 0.5, 0.5)
editor.add_vertex(3, 0.0, 0.5)
editor.add_vertex(4, -0.5, 0.5)
editor.add_vertex(5, -0.5, 0.0)
editor.add_vertex(6, -0.5, -0.5)
editor.add_vertex(7, 0.0, -0.5)
editor.add_vertex(8, 0.5, -0.5)
editor.add_vertex(9, -0.5, 0.0)
editor.add_cell(0, 0, 1, 3)
editor.add_cell(1, 1, 2, 3)
editor.add_cell(2, 0, 3, 4)
editor.add_cell(3, 0, 4, 5)
editor.add_cell(4, 0, 9, 6)
editor.add_cell(5, 0, 6, 7)
editor.add_cell(6, 0, 7, 8)
editor.add_cell(7, 0, 8, 1)
editor.close()
num_refine = 6
#print num_refine
h=0.5**num_refine
for i in range(num_refine):
    mesh=refine(mesh)
n=1+pow(2,num_refine+1)
#print n
#plot(mesh)
#interactive()

V = VectorFunctionSpace(mesh ,"Lagrange",1)
W = FunctionSpace(mesh,"Lagrange",1)
#================================================================================================#

#-------------------- Define material property----------------------------#
#E = 200e9     
#nu = 0.3
#lmbda = E*nu /((1.0 + nu)*(1.0-2*nu)) 
#mu = E / (2 * (1 + nu))
lmbda=121.15E3
mu=80.77E3
Gc = 2.7
l0=h
#print l0
k=1E-6

def  epsilon(y):
	D = nabla_grad(y)
	return  0.5 * (D+D.T)
def  sigma(y):
	return 2 * mu * epsilon(y) + lmbda * tr(epsilon(y)) * Identity (2)
def  str_egy_ele(y):
	str_ele = epsilon(y)
	trace_str = tr(str_ele)
	trace_str_sq = tr(str_ele*str_ele)
	return (0.5*lmbda*trace_str**2) + mu*trace_str_sq
#===============================================================================================#

#----------------------Define boundary conditions and mark boundaries----------------------------#
tol=1E-14
class bottom(SubDomain):
	def inside(self,x,on_boundary):
		return abs(x[1]+0.5)<=tol and on_boundary

class top(SubDomain):
	def inside(self,x,on_boundary):
		return abs(x[1]-0.5)<=tol and on_boundary
class middle(SubDomain):
	def inside(self,x,on_boundary):
		return x[1]==0.0 and x[0]<=0.0

middle=middle()
bottom=bottom()
top=top()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
bottom.mark(boundaries,1)
top.mark(boundaries,2)

u0=Constant(("0.0","0.0"))
bc1=DirichletBC(V,u0,boundaries,1)

p0=Constant(1.0)
bc2=DirichletBC(W,p0,middle)

u_top = Expression(("0.0","t"),t=0.0)
bc3=DirichletBC(V,u_top,boundaries,2)

bc=[bc1,bc3]
#===============================================================================================#

#---------------------------- Define variational problem----------------------------------------#
u = TrialFunction(V)
v = TestFunction(V)

u_new = Function(V)
u_old = Function(V)
u_conv = Function(V)

phi = TrialFunction(W)
w = TestFunction(W)

phi_old = initial_notch(mesh,l0)
phi_new = Function(W)
phi_conv = Function(W)

disp_eq = ((1-phi_old)**2+k) * inner(nabla_grad(v), sigma(u))*dx

phi_evo = Gc*l0*inner(nabla_grad(phi),nabla_grad(w))*dx +\
            ((Gc/l0) + 2.*str_egy_ele(u_new))*inner(phi,w)*dx-\
            2.*str_egy_ele(u_new)*w*dx
#==============================================================================================#

#---------------------------- compute solutions-----------------------------------------------#
problem_disp_eq = LinearVariationalProblem(lhs(disp_eq),rhs(disp_eq),u_new,bc)
solver=LinearVariationalSolver(problem_disp_eq)

problem_phi_evo = LinearVariationalProblem(lhs(phi_evo),rhs(phi_evo),phi_new,bc2)
solver1 = LinearVariationalSolver(problem_phi_evo)

min_disp = 0.0
max_disp = 0.006
steps=6000
load_multipliers = np.linspace(min_disp,max_disp,steps)

f_allY_s=[]
f_x_u=[]
for (i_t,t) in enumerate(load_multipliers):
	u_top.t=t
	#print(u_top.t)
   	u_old.assign(u_conv)
	phi_old.assign(phi_conv)	
	
	toll = 0.01
	err = 1.0
	
	iter = 1
	maxiter = 100
	while err > toll:
        	solver.solve()
		solver1.solve()
		
		u_dispdiff = u_new.vector().array() - u_old.vector().array()      	
		err_u= np.linalg.norm(u_dispdiff, ord=2)
		phi_diff = phi_new.vector().array() - phi_old.vector().array()
        	err_phi= np.linalg.norm(phi_diff, ord=2)
		err = max(err_u,err_phi)
		u_old.assign(u_new)
		phi_old.assign(phi_new)	
        	iter = iter + 1
   	    	if err < toll:
           		print "Solution Converge after",iter,"Error",err
          		
			u_conv.assign(u_new)
			phi_conv.assign(phi_new)
			
			k_matx=assemble(disp_eq)
			u_vec=u_new.vector()
			F_vec=k_matx*u_vec		
			
			F_vec_func=Function(V,F_vec)		
			f_allY=[]	
					
			for k in np.linspace(-0.5,0.5,n):	  			
				bottom_force= F_vec_func(k,-0.5)       #force vector at y=0.5 and x=k 
				y_direc_force=bottom_force[1]			#y direction force at y=0.5 and x=k 				
				f_allY.append(y_direc_force)				
				array_y_bottom_force=f_allY		# ARRAY OF ALL y direction force at y=0.5	
			b=sum(array_y_bottom_force)						
			f_x_u.append(b)	
			plt.scatter(u_top.t,np.negative(b))
			plt.pause(0.05)		
			
         		plot(u_new,key = 'u',title = 'u%.4f'%(t),mode='displacement',axes=True)
			q=plot(phi_new,range_min = 0.,range_max = 1.,key = 'alpha',title = 'alpha%.4f'%(t),axes=True)	
			#file = File("phi%.4f.pvd"%(t))
			#file << phi_new
interactive()
#==========================================================================================#
plt.close() 
x = load_multipliers
y2=np.negative(f_x_u)
plt.plot(x,y2)
plt.show()

'''
fl = open('fval.csv', 'w')

writer = csv.writer(fl)
for values in f_y_u:
    writer.writerow(values)

fl.close()'''  
