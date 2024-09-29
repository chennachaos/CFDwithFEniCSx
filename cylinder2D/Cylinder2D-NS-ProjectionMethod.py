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

from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc


# specific functions from dolfinx modules
from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression, form, assemble_scalar )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
#from dolfinx.io import VTXWriter

# specific functions from ufl modules
import ufl
from ufl import (TestFunction, TrialFunction, Identity, nabla_grad, lhs, rhs, grad, sym, det, div, dev, inv, tr, sqrt, conditional ,\
                 gt, dx, ds, inner, derivative, dot, ln, split, exp, eq, cos, acos, ge, le, FacetNormal, as_vector)

# basix finite elements
import basix
from basix.ufl import element, mixed_element, quadrature_element




msh, cell_tags, facet_tags = io.gmshio.read_from_msh("CFD-cylinder-P2.msh", MPI.COMM_WORLD)

# coordinates of the nodes
x = ufl.SpatialCoordinate(msh)

normals = -FacetNormal(msh)  # Normal pointing out of obstacle


# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx = ufl.Measure('dx', domain=msh, subdomain_data=cell_tags,  metadata={'quadrature_degree': 4})
ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 4})



# FE Elements
# Quadratic element for velocity and linear element for pressure
###
deg_u = 2
deg_p = deg_u-1

# velocity
V2 = element("Lagrange", msh.basix_cell(), deg_u, shape=(msh.geometry.dim,))
# pressure
P1 = element("Lagrange", msh.basix_cell(), deg_p)

# functions spaces
V = functionspace(msh, V2)
Q = functionspace(msh, P1)

# trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

p = TrialFunction(Q)
q = TestFunction(Q)

# functions with DOFs at the current step

# functions with DOFs at the previous step
u_n = Function(V)
u_n.name = "u_n"
p_n = Function(Q)
p_n.name = "p_n"


# strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# stress tensor
def sigma(u,p):
    return 2*mu*epsilon(u)-p*Identity(len(u))



t = 0.0
dt = 0.01
time_final = 400.0#dt*1000

num_steps = np.int32(time_final/dt) + 1

# Parameter values
rho = Constant(msh, PETSc.ScalarType(1.0))
bforce = Constant(msh, PETSc.ScalarType( (0.0,0.0,0.0) ))

mu  = Constant(msh, PETSc.ScalarType(0.01))

dtk = Constant(msh, PETSc.ScalarType(dt))


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

dofs_1_x = fem.locate_dofs_topological(V.sub(0), facet_tags.dim, facet_tags.find(1)) #vx
dofs_1_y = fem.locate_dofs_topological(V.sub(1), facet_tags.dim, facet_tags.find(1)) #vy

dofs_3_y = fem.locate_dofs_topological(V.sub(1), facet_tags.dim, facet_tags.find(3)) #uy
dofs_4_y = fem.locate_dofs_topological(V.sub(1), facet_tags.dim, facet_tags.find(4)) #uy

dofs_5_x = fem.locate_dofs_topological(V.sub(0), facet_tags.dim, facet_tags.find(5)) #ux
dofs_5_y = fem.locate_dofs_topological(V.sub(1), facet_tags.dim, facet_tags.find(5)) #uy


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


bc1_x = fem.dirichletbc(value=1.0, dofs=dofs_1_x, V=V.sub(0))
bc1_y = fem.dirichletbc(value=0.0, dofs=dofs_1_y, V=V.sub(1))

bc3_y = fem.dirichletbc(value=0.0, dofs=dofs_3_y, V=V.sub(1))
bc4_y = fem.dirichletbc(value=0.0, dofs=dofs_4_y, V=V.sub(1))

bc5_x = fem.dirichletbc(value=0.0, dofs=dofs_5_x, V=V.sub(0))
bc5_y = fem.dirichletbc(value=0.0, dofs=dofs_5_y, V=V.sub(1))


bcu = [bc1_x, bc1_y, bc3_y, bc4_y, bc5_x, bc5_y]



# Pressure BCs

dofs_2_p = fem.locate_dofs_topological(Q, facet_tags.dim, facet_tags.find(2))
bc2_p    = fem.dirichletbc(value=0.0, dofs=dofs_2_p, V=Q)


bcp = [bc2_p]




# dimension
d = len(u)

# Kinematics
Id = Identity(d)

# average velocity
U = 0.5*(u+u_n)

#
#
# Variational problem for Step 1
#
F1  = rho*dot( (u-u_n)/dtk ,v)*dx
F1 += rho*dot( dot(u_n,nabla_grad(u_n)) ,v)*dx
F1 += inner(sigma(U, p_n), epsilon(v))*dx
F1 += dot(p_n*normals, v)*ds - dot(mu*nabla_grad(U)*normals, v)*ds
F1 -= dot(bforce,v)*dx

a1 = form(lhs(F1))
L1 = form(rhs(F1))

A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = create_vector(L1)



# Define variational problem for Step 2
u_ = Function(V)
a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho/dtk) * div(u_) * q * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)


# Define variational problem for Step 3
p_ = Function(Q)
a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_, v) * dx - dtk * dot(nabla_grad(p_ - p_n), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)




# Solver for step 1
solver1 = PETSc.KSP().create(msh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for step 2
solver2 = PETSc.KSP().create(msh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.BCGS)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(msh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)



obstacle_marker = 5
dObs = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=obstacle_marker)

drag = form( ( (mu*grad(u_)[0,0]-p_)*normals[0] + (mu*grad(u_)[0,1])*normals[1] )   * dObs)
lift = form( ( (mu*grad(u_)[1,0])*normals[0]   + (mu*grad(u_)[1,1]-p_)*normals[1] ) * dObs)
    

C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)




# displacement projection
U1 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))

Vs = functionspace(msh, U1)

u_proj = Function(Vs)
u_proj.name = "velocity"

p_proj = Function(functionspace(msh,P1))
p_proj.name = "pressure"


fname = "./results-ProjMethod/cylinder2d-velo-.pvd"
VTKfile_Velo = io.VTKFile(msh.comm, fname, "w")
VTKfile_Velo.write_mesh(msh)

fname = "./results-ProjMethod/cylinder2d-pres-.pvd"
VTKfile_Pres = io.VTKFile(msh.comm, fname, "w")
VTKfile_Pres.write_mesh(msh)

# function to write results to XDMF at time t
def writeResults_Velo(time_cur, timeStep):

    u_proj.interpolate(u_)

    #    file.close()
    VTKfile_Velo.write_function(u_proj)

# function to write results to XDMF at time t
def writeResults_Pres(time_cur, timeStep):
    
    p_proj.interpolate(p_)

    VTKfile_Pres.write_function(p_proj)


fname = "Cylinder2D-Re100-forces-projection-method.dat"
file_forces = open(fname,"w")


print("\n----------------------------\n")
print("Simulation has started")
print("\n----------------------------\n")

timeStep = 0
time_cur = 0.0

drag_coeff = 0.0
lift_coeff = 0.0

writeResults_Velo(time_cur, timeStep)
writeResults_Pres(time_cur, timeStep)
print(time_cur, 0.0, 0.0, file=file_forces)



#for i in range(num_steps):
while ( round(time_cur+dt, 6) <= time_final):
    # Update current time step
    t += dt
    time_cur += dt
    timeStep += 1

    print("\n\n Load step = ", timeStep)
    print("     Time      = ", time_cur)

    # Step 1: Tentative veolcity step
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()
    # Update variable with solution form this time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    #drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
    #lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
    drag_coeff = assemble_scalar(drag)
    lift_coeff = assemble_scalar(lift)
    
    C_D[timeStep] = drag_coeff
    C_L[timeStep] = lift_coeff
    print(time_cur, drag_coeff, lift_coeff, file=file_forces)

    # Write solutions to file
    if (timeStep % 10 == 0):
        #    print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")
        writeResults_Velo(time_cur, timeStep)
        writeResults_Pres(time_cur, timeStep)
        print(drag_coeff, lift_coeff)

    # Compute error at current time-step
    #error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
    #error_max = mesh.comm.allreduce(np.max(u_.vector.array - u_ex.vector.array), op=MPI.MAX)
    # Print error only every 20th step and at the last step

# Close files
VTKfile_Velo.close()
VTKfile_Pres.close()
file_forces.close()

# Destroy solvers
b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()

