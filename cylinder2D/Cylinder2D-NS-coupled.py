"""
@Problem: Flow past a fixed circular cylinder in 2D.

@Formulation: Mixed velocity-pressure formulation.

@author: Dr Chennakesava Kadapa

Created on Sun 24-Jun-2024
"""


# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# For MPI-based parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# PETSc solvers
from petsc4py import PETSc

# specific functions from dolfinx modules
from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression, form, assemble_scalar )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
#from dolfinx.io import VTXWriter

# specific functions from ufl modules
import ufl
from ufl import (TestFunctions, TrialFunction, Identity, nabla_grad, grad, sym, det, div, dev, inv, tr, sqrt, conditional ,\
                 gt, dx, ds, inner, derivative, dot, ln, split, exp, eq, cos, acos, ge, le, FacetNormal, as_vector)

# basix finite elements
import basix
from basix.ufl import element, mixed_element, quadrature_element




msh, markers, facet_tags = io.gmshio.read_from_msh("CFD-cylinder-P1.msh", MPI.COMM_WORLD)

# coordinates of the nodes
x = ufl.SpatialCoordinate(msh)


# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 4})
ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 4})



# FE Elements
# Quadratic element for displacement
###
###
# Define function spaces and elements
deg_u = 2
deg_p = deg_u-1

# displacement
U2 = element("Lagrange", msh.basix_cell(), deg_u, shape=(msh.geometry.dim,))
# pressure
P1 = element("Lagrange", msh.basix_cell(), deg_p)

# Mixed element
TH = mixed_element([U2, P1])
ME = functionspace(msh, TH)


V2 = functionspace(msh, U2) # Vector function space


# functions with DOFs at the current step
w = Function(ME)
u, p = split(w)

# functions with DOFs at the current step
w_new = Function(ME)
u_new, p_new = split(w_new)

# functions with DOFs at the previous step
w_old = Function(ME)
u_old, p_old = split(w_old)

# functions with DOFs at the previous step
w_old2 = Function(ME)
u_old2, p_old2 = split(w_old2)

# Test functions
u_test, p_test = TestFunctions(ME)

# current acceleration
a_new = Function(V2)

# old acceleration
a_old = Function(V2)


# Trial functions
dw = TrialFunction(ME)


# Parameter values
rho = Constant(msh, PETSc.ScalarType(1.0))
mu  = Constant(msh, PETSc.ScalarType(0.01))


t = 0.0
dt = 0.1
time_final = 300.0

num_steps = np.int32(time_final/dt) + 1


# time integration parameters
# generalised-alpha method
specRad = 0.0
alpf = 1.0/(1.0+specRad)
alpm = 0.5*(3.0-specRad)/(1.0+specRad)
gamm = 0.5+alpm-alpf


# Boundary conditions

'''
1 1 "inlet"
1 2 "outlet"
1 3 "topedge"
1 4 "bottomedge"
1 5 "cylinder"
2 6 "fluid"
'''

dofs_1_x = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(1)) #vx
dofs_1_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(1)) #vy

dofs_3_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(3)) #uy
dofs_4_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(4)) #uy

dofs_5_x = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(5)) #ux
dofs_5_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(5)) #uy


class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        #values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        #values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
        values = 1.0
        return values


# Inlet
#u_inlet = Function(V2)
#inlet_velocity = InletVelocity(t)
#u_inlet.interpolate(inlet_velocity)
#bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
# Walls


bc1_x = fem.dirichletbc(value=1.0, dofs=dofs_1_x, V=ME.sub(0).sub(0))
bc1_y = fem.dirichletbc(value=0.0, dofs=dofs_1_y, V=ME.sub(0).sub(1))

bc3_y = fem.dirichletbc(value=0.0, dofs=dofs_3_y, V=ME.sub(0).sub(1))
bc4_y = fem.dirichletbc(value=0.0, dofs=dofs_4_y, V=ME.sub(0).sub(1))

bc5_x = fem.dirichletbc(value=0.0, dofs=dofs_5_x, V=ME.sub(0).sub(0))
bc5_y = fem.dirichletbc(value=0.0, dofs=dofs_5_y, V=ME.sub(0).sub(1))


bcs = [bc1_x, bc1_y, bc3_y, bc4_y, bc5_x, bc5_y]




# dimension
d = len(u)

# Kinematics
Id = Identity(d)


u_avg = alpf*u + (1.0-alpf)*u_old
p_avg = alpf*p + (1.0-alpf)*p_old

a = (1.0/gamm/dt)*(u-u_old) + ((gamm-1.0)/gamm)*a_old

# expression for copying/storing the values
a_expr = Expression(a, V2.element.interpolation_points())

a_avg = alpm*a + (1.0-alpm)*a_old



# Weak form
#Res = rho*inner(dot(2*u_old-u_old2, nabla_grad(u)), u_test)*dx + mu*inner(grad(u), grad(u_test))*dx - inner(p, div(u_test))*dx + inner(div(u), p_test)*dx

Res =  rho*inner(a_avg, u_test)*dx
Res += rho*inner(dot(u_old, nabla_grad(u_avg)), u_test)*dx 
Res += rho*inner(dot(u_avg, nabla_grad(u_old)), u_test)*dx 
Res -= rho*inner(dot(u_old, nabla_grad(u_old)), u_test)*dx 

#Res =  rho*inner(dot(2*u_old-u_old2, nabla_grad(u_avg)), u_test)*dx
#Res =  rho*inner(dot(u_old, nabla_grad(u_avg)), u_test)*dx
Res += mu*inner(grad(u_avg), grad(u_test))*dx
Res -= inner(p_avg, div(u_test))*dx
Res -= inner(div(u_avg), p_test)*dx


dRes = derivative(Res, w, dw)


# set up the nonlinear problem
problem = NonlinearProblem(Res, w, bcs, dRes)



# set the solver parameters
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.report = True


#  The Krylov solver parameters.
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()




normals = -FacetNormal(msh)  # Normal pointing out of obstacle
obstacle_marker = 5
dObs = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=obstacle_marker)

drag = form( ( (mu*grad(u)[0,0]-p)*normals[0] + (mu*grad(u)[0,1])*normals[1] )   * dObs)
lift = form( ( (mu*grad(u)[1,0])*normals[0]   + (mu*grad(u)[1,1]-p)*normals[1] ) * dObs)


C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)




# displacement projection
U1 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))

Vs = functionspace(msh, U1)

u_proj = Function(Vs)
u_proj.name = "velocity"
u_proj.interpolate(w_new.sub(0))

p_proj = Function(functionspace(msh,P1))
p_proj.name = "pressure"
p_proj.interpolate(w_new.sub(1))
#p_proj = w.sub(1)


fname = "cylinder2d-velo-.pvd"
VTKfile_Velo = io.VTKFile(msh.comm, fname, "w")
VTKfile_Velo.write_mesh(msh)

fname = "cylinder2d-pres-.pvd"
VTKfile_Pres = io.VTKFile(msh.comm, fname, "w")
VTKfile_Pres.write_mesh(msh)

# function to write results to XDMF at time t
def writeResults_Velo(time_cur, timeStep):

    u_proj.interpolate(w_new.sub(0))

    #    file.close()
    VTKfile_Velo.write_function(u_proj)

# function to write results to XDMF at time t
def writeResults_Pres(time_cur, timeStep):
    
    p_proj.interpolate(w_new.sub(1))

    VTKfile_Pres.write_function(p_proj)


fname = "Cylinder2D-Re100-forces.dat"
file_forces = open(fname,"w")


print("\n----------------------------\n")
print("Simulation has started")
print("\n----------------------------\n")

timeStep = 0
time_cur = 0.0

writeResults_Velo(time_cur, timeStep)
writeResults_Pres(time_cur, timeStep)
print(time_cur, 0.0, 0.0, file=file_forces)


# loop over time steps
#for timeStep in range(num_steps):
while ( round(time_cur+dt, 6) <= time_final):
    # Update current time
    time_cur += dt
    timeStep += 1

    print("\n\n Load step = ", timeStep)
    print("     Time      = ", time_cur)

    #tz.value = -320*time_cur
    #traction.value = time_cur
    #traction.value = (0.0,0.0,-320.0*time_cur)
    #traction = Constant(msh, (0.0,0.625*time_cur,0.0))

    # Solve the problem
    # Compute solution
    (num_its, converged) = solver.solve(w)

    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged.")


    # compute new acceleration
    a_new.interpolate(a_expr)


    writeResults_Velo(time_cur, timeStep)
    writeResults_Pres(time_cur, timeStep)

    # save variables
    # velocity
    #w_old2.x.array[:] = w_old.x.array
    #w_old.x.array[:] = w.x.array

    w_old.vector.copy(w_old2.vector)
    w.vector.copy(w_old.vector)

    # acceleration
    a_new.vector.copy(a_old.vector)
    #a_old.x.array[:] = a_temp.x.array[:]

    #drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
    #lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
    drag_coeff = assemble_scalar(drag)
    lift_coeff = assemble_scalar(lift)
    
    print(drag_coeff, lift_coeff)

    C_D[timeStep] = drag_coeff
    C_L[timeStep] = lift_coeff
    print(time_cur, drag_coeff, lift_coeff, file=file_forces)


VTKfile_Velo.close()
VTKfile_Pres.close()
file_forces.close()
