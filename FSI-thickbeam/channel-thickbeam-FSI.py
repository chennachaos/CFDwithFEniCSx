"""
@Problem: Flow past a circular cylinder in 2D with prescribed motion.

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
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression, form, assemble, assemble_scalar )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_matrix, create_vector, set_bc
from dolfinx.io import XDMFFile
#from dolfinx.io import VTXWriter

# specific functions from ufl modules
import ufl
from ufl import (TestFunction, TestFunctions, TrialFunction, Identity, lhs, rhs, nabla_grad, grad, sym, det, div, dev, inv, tr, sqrt, conditional , CellDiameter, \
                 gt, dx, ds, inner, derivative, dot, ln, split, exp, eq, cos, acos, ge, le, FacetNormal, as_vector)

# basix finite elements
import basix
from basix.ufl import element, mixed_element, quadrature_element




domain_fluid, domain_fluid_markers, domain_fluid_facet_tags = io.gmshio.read_from_msh("channel-thickbeam-2D-fluid-P2.msh", MPI.COMM_WORLD)

# coordinates of the nodes
coords_fluid = ufl.SpatialCoordinate(domain_fluid)

# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx_fluid = ufl.Measure('dx', domain=domain_fluid, metadata={'quadrature_degree': 4})
ds_fluid = ufl.Measure('ds', domain=domain_fluid, subdomain_data=domain_fluid_facet_tags, metadata={'quadrature_degree': 4})



domain_solid, domain_solid_markers, domain_solid_facet_tags = io.gmshio.read_from_msh("channel-thickbeam-2D-solid-P2.msh", MPI.COMM_WORLD)

# coordinates of the nodes
coords_solid = ufl.SpatialCoordinate(domain_solid)

# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx_solid = ufl.Measure('dx', domain=domain_solid, metadata={'quadrature_degree': 4})
ds_solid = ufl.Measure('ds', domain=domain_solid, subdomain_data=domain_solid_facet_tags, metadata={'quadrature_degree': 4})


t = 0.0
dt = 0.02
time_final = dt*10

num_steps = np.int32(time_final/dt) + 1

dtks = Constant(domain_solid, PETSc.ScalarType(dt))
dtkf = Constant(domain_fluid, PETSc.ScalarType(dt))

'''
1 1 "bottomedge"
1 2 "interface"
2 3 "solid"
'''

deg_u = 2
# displacement
U2_solid = element("Lagrange", domain_solid.basix_cell(), deg_u, shape=(domain_solid.geometry.dim,))

V2_solid = functionspace(domain_solid, U2_solid) # Vector function space

## trial and test functions for the solid problem
solid_disp_trial = TrialFunction(V2_solid)
solid_disp_test  = TestFunction(V2_solid)

solid_disp           = Function(V2_solid)
solid_disp_old       = Function(V2_solid)
solid_disp_total     = Function(V2_solid)
solid_disp_total_old = Function(V2_solid)

# solid velocities
solid_velo     = Function(V2_solid)
solid_velo_old = Function(V2_solid)

# solid accelerations
solid_acce     = Function(V2_solid)
solid_acce_old = Function(V2_solid)

# solid forces
solid_force      = Function(V2_solid)
solid_force_old  = Function(V2_solid)
solid_force_old2 = Function(V2_solid)


solid_dofs_1_x = fem.locate_dofs_topological(V2_solid.sub(0), domain_solid_facet_tags.dim, domain_solid_facet_tags.find(1)) #vx
solid_dofs_1_y = fem.locate_dofs_topological(V2_solid.sub(1), domain_solid_facet_tags.dim, domain_solid_facet_tags.find(1)) #vy

solid_dofs_2_x = fem.locate_dofs_topological(V2_solid.sub(0), domain_solid_facet_tags.dim, domain_solid_facet_tags.find(2)) #vx
solid_dofs_2_y = fem.locate_dofs_topological(V2_solid.sub(1), domain_solid_facet_tags.dim, domain_solid_facet_tags.find(2)) #vy


solid_bc1_x = fem.dirichletbc(value=0.0, dofs=solid_dofs_1_x, V=V2_solid.sub(0))
solid_bc1_y = fem.dirichletbc(value=0.0, dofs=solid_dofs_1_y, V=V2_solid.sub(1))

solid_bc2_x = fem.dirichletbc(value=0.0, dofs=solid_dofs_2_x, V=V2_solid.sub(0))
solid_bc2_y = fem.dirichletbc(value=0.0, dofs=solid_dofs_2_y, V=V2_solid.sub(1))

#motionY = fem.Constant(domain_fluid, PETSc.ScalarType(0.0))
#meshmotion_bc5_y = fem.dirichletbc(value=motionY, dofs=meshmotion_dofs_5_y, V=V2.sub(1))

bcs_solid = [solid_bc1_x, solid_bc1_y, solid_bc2_x, solid_bc2_y]


# constituive relations - linear elasticity
rho_solid = fem.Constant(domain_solid, PETSc.ScalarType(10.0))
E_solid   = fem.Constant(domain_solid, PETSc.ScalarType(200.0))
nu_solid  = fem.Constant(domain_solid, PETSc.ScalarType(0.3))

mu_solid  = fem.Constant(domain_solid, PETSc.ScalarType(E_solid/2.0/(1+nu_solid) ))
K_solid   = fem.Constant(domain_solid, PETSc.ScalarType(E_solid/3.0/(1-2.0*nu_solid) ))
lambda_solid = fem.Constant(domain_solid, PETSc.ScalarType(K_solid-2.0*mu_solid/3.0 ))


# parameters of the time integration scheme
specRad_s = Constant(domain_solid, PETSc.ScalarType(0.0)) 
af_s      = Constant(domain_solid, PETSc.ScalarType( 1.0/(1.0+specRad_s) ))
am_s      = Constant(domain_solid, PETSc.ScalarType( (2.0-specRad_s)/(1.0+specRad_s) ))
gamma_s   = Constant(domain_solid, PETSc.ScalarType( 0.5+am_s-af_s ))
beta_s    = Constant(domain_solid, PETSc.ScalarType( 0.25*(1.0+am_s-af_s)**2 ))


#def eps(v):
#    return sym(grad(v))

#def sigma(v):
#    return lambda_mesh*tr(eps(v))*Identity(2) + 2.0*mu_mesh*eps(v)

Id = Identity(3)

solid_disp_avg = af_s*solid_disp_trial + (1.0-af_s)*solid_disp_old

# small-strain tensor
epsilon_solid = 0.5*(grad(solid_disp_avg) + grad(solid_disp_avg).T)

# volumetric strain
epsvol_solid = tr(epsilon_solid)

# stress tensor
sigma_solid  = 2*mu_solid*epsilon_solid + lambda_solid*epsvol_solid*Id


solid_velo_formula = (gamma_s/beta_s/dtks)*(solid_disp - solid_disp_old) + (1.0-gamma_s/beta_s)*solid_velo_old + (1.0-gamma_s/2.0/beta_s)*dtks*solid_acce_old

solid_acce_formula = (1.0/beta_s/dtks/dtks)*(solid_disp - solid_disp_old) - (1.0/beta_s/dtks)*solid_velo_old - (1.0/2.0/beta_s-1.0)*solid_acce_old


# expression for copying/storing the values
solid_velo_expr = Expression(solid_velo_formula, V2_solid.element.interpolation_points())
solid_acce_expr = Expression(solid_acce_formula, V2_solid.element.interpolation_points())


solid_acce_new = (1.0/beta_s/dt/dt)*(solid_disp_trial - solid_disp_old) - (1.0/beta_s/dt)*solid_velo_old - (1.0/2.0/beta_s-1.0)*solid_acce_old

solid_acce_avg = am_s*solid_acce_new + (1.0-am_s)*solid_acce_old

Res_solid = inner(sigma_solid, sym(grad(solid_disp_test)) )*dx_solid
#Res_solid  = rho_solid*inner(solid_acce_avg, solid_disp_test)*dx_solid
#


#dRes_solid = derivative(Res_solid, solid_disp, solid_disp_trial)

# set up the nonlinear problem
#problem_solid = NonlinearProblem(Res_solid, solid_disp, bcs_solid, dRes_solid)


a_solid = form(lhs(Res_solid))
L_solid = form(rhs(Res_solid))

#A_solid = assemble_matrix(a_solid, bcs=bcs_solid)
#A_solid.assemble()
A_solid = create_matrix(a_solid)
b_solid = create_vector(L_solid)


# set the solver parameters
solver_solid = PETSc.KSP().create(domain_solid.comm)
solver_solid.setOperators(A_solid)
solver_solid.setType(PETSc.KSP.Type.PREONLY)
pc_solid = solver_solid.getPC()
pc_solid.setType(PETSc.PC.Type.LU)

#solver_solid = NewtonSolver(MPI.COMM_WORLD, problem_solid)
#solver_solid.convergence_criterion = "incremental"
solver_solid.rtol = 1e-8
solver_solid.atol = 1e-8
solver_solid.max_it = 50
#solver_solid.report = True









# FE Elements
# Quadratic element for displacement
###
###
# Define function spaces and elements
deg_u = 2
deg_p = deg_u-1

# displacement
U2 = element("Lagrange", domain_fluid.basix_cell(), deg_u, shape=(domain_fluid.geometry.dim,))
# pressure
P1 = element("Lagrange", domain_fluid.basix_cell(), deg_p)

V2 = functionspace(domain_fluid, U2) # Vector function space


## variables for the mesh motion problem
mesh_disp      = Function(V2)

mesh_disp_test  = TestFunction(V2)
mesh_disp_trial = TrialFunction(V2)

mesh_disp_total   = Function(V2)
mesh_disp_total_old = Function(V2)

## variables for the fluid problem
# Mixed element
TH = mixed_element([U2, P1])
ME = functionspace(domain_fluid, TH)



# functions with DOFs at the current step
w = Function(ME)
fluid_velo, fluid_pres = split(w)

# functions with DOFs at the previous step
w_old = Function(ME)
fluid_velo_old, fluid_pres_old = split(w_old)

# functions with DOFs at the previous step
w_old2 = Function(ME)
fluid_velo_old2, fluid_pres_old2 = split(w_old2)

# Test functions
fluid_velo_test, fluid_pres_test = TestFunctions(ME)

# mesh velocities
mesh_velo = Function(V2)
mesh_velo_old = Function(V2)

# current acceleration
fluid_acce = Function(V2)

# old acceleration
fluid_acce_old = Function(V2)


# Trial functions
dw = TrialFunction(ME)


# Parameter values
rho = Constant(domain_fluid, PETSc.ScalarType(1.0))
mu  = Constant(domain_fluid, PETSc.ScalarType(0.05))


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
1 3 "bottomedge"
1 4 "topedge"
1 5 "interface"
2 6 "fluid"
'''

meshmotion_dofs_1_x = fem.locate_dofs_topological(V2.sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(1)) #vx
meshmotion_dofs_1_y = fem.locate_dofs_topological(V2.sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(1)) #vy

meshmotion_dofs_2_x = fem.locate_dofs_topological(V2.sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(2)) #vx
meshmotion_dofs_2_y = fem.locate_dofs_topological(V2.sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(2)) #vy

meshmotion_dofs_3_x = fem.locate_dofs_topological(V2.sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(3)) #ux
meshmotion_dofs_3_y = fem.locate_dofs_topological(V2.sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(3)) #uy

meshmotion_dofs_4_x = fem.locate_dofs_topological(V2.sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(4)) #ux
meshmotion_dofs_4_y = fem.locate_dofs_topological(V2.sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(4)) #uy

meshmotion_dofs_5_x = fem.locate_dofs_topological(V2.sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(5)) #ux
meshmotion_dofs_5_y = fem.locate_dofs_topological(V2.sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(5)) #uy


class CylinderMotion():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        #values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
        fs = 0.1667 # shedding frequency at Re=100
        F  = 0.8
        f0 = F*fs  # cylinder oscillating frequency
        A  = 0.1   # non-dimensional amplitude
        values[1] = A*np.sin(2.0*np.pi*f0*t)
        return values


#motionY_func = Function(V2)
#motionY = CylinderMotion(t)
#motionY_func.interpolate(motionY)
#bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
# Walls


meshmotion_bc1_x = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_1_x, V=V2.sub(0))
meshmotion_bc1_y = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_1_y, V=V2.sub(1))

meshmotion_bc2_x = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_2_x, V=V2.sub(0))
meshmotion_bc2_y = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_2_y, V=V2.sub(1))

meshmotion_bc3_x = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_3_x, V=V2.sub(0))
meshmotion_bc3_y = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_3_y, V=V2.sub(1))

meshmotion_bc4_x = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_4_x, V=V2.sub(0))
meshmotion_bc4_y = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_4_y, V=V2.sub(1))

meshmotion_bc5_x = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_5_x, V=V2.sub(0))
#meshmotion_bc5_y = fem.dirichletbc(value=0.0, dofs=meshmotion_dofs_5_y, V=ME.sub(0).sub(1))

motionY = fem.Constant(domain_fluid, PETSc.ScalarType(0.0))

meshmotion_bc5_y = fem.dirichletbc(value=motionY, dofs=meshmotion_dofs_5_y, V=V2.sub(1))

bcs_meshmotion = [meshmotion_bc1_x, meshmotion_bc1_y, meshmotion_bc2_x, meshmotion_bc2_y, meshmotion_bc3_x, meshmotion_bc3_y, meshmotion_bc4_x, meshmotion_bc4_y, meshmotion_bc5_x, meshmotion_bc5_y]
#bcs = []


# constituive relations - linear elasticity
E_mesh  = fem.Constant(domain_fluid, PETSc.ScalarType(1.0))
nu_mesh = fem.Constant(domain_fluid, PETSc.ScalarType(0.1))

mu_mesh = E_mesh/2.0/(1+nu_mesh)
K_mesh  = E_mesh/3.0/(1-2.0*nu_mesh)
lambda_mesh = K_mesh-2.0*mu_mesh/3.0


detJ = abs(ufl.JacobianDeterminant(domain_fluid))

#def eps(v):
#    return sym(grad(v))

#def sigma(v):
#    return lambda_mesh*tr(eps(v))*Identity(2) + 2.0*mu_mesh*eps(v)

Id = Identity(3)

# small-strain tensor
epsilon = 0.5*(grad(mesh_disp) + grad(mesh_disp).T)

# volumetric strain
epsvol = tr(epsilon)

#elemRad = Circumradius(domain_fluid)

# stress tensor
#sigma_mesh  = 2*mu_mesh*(1.0/elemRad**2)*epsilon + lambda_mesh*epsvol*Id
sigma_mesh  = 2*mu_mesh*epsilon + lambda_mesh*epsvol*Id

#Res = inner(sigma, 0.5*(grad(mesh_disp_test)+grad(mesh_disp_test).T) )*dx_fluid
#Res = inner(sigma(mesh_disp), eps(mesh_disp_test) )*dx_fluid

#Res_mesh = inner(sigma_mesh, sym(grad(mesh_disp_test)) )*dx_fluid
Res_mesh = inner(sigma_mesh*detJ**(-2), sym(grad(mesh_disp_test)) )*dx_fluid


dRes_mesh = derivative(Res_mesh, mesh_disp, mesh_disp_trial)


# set up the nonlinear problem
problem_meshmotion = NonlinearProblem(Res_mesh, mesh_disp, bcs_meshmotion, dRes_mesh)


# set the solver parameters
solver_meshmotion = NewtonSolver(MPI.COMM_WORLD, problem_meshmotion)
solver_meshmotion.convergence_criterion = "incremental"
solver_meshmotion.rtol = 1e-8
solver_meshmotion.atol = 1e-8
solver_meshmotion.max_it = 50
solver_meshmotion.report = True


ksp = solver_meshmotion.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()


## Flow problem

dofs_1_x = fem.locate_dofs_topological(ME.sub(0).sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(1)) #vx
dofs_1_y = fem.locate_dofs_topological(ME.sub(0).sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(1)) #vy

dofs_3_x = fem.locate_dofs_topological(ME.sub(0).sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(3)) #ux
dofs_3_y = fem.locate_dofs_topological(ME.sub(0).sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(3)) #uy

dofs_4_x = fem.locate_dofs_topological(ME.sub(0).sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(4)) #ux
dofs_4_y = fem.locate_dofs_topological(ME.sub(0).sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(4)) #uy

dofs_5_x = fem.locate_dofs_topological(ME.sub(0).sub(0), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(5)) #ux
dofs_5_y = fem.locate_dofs_topological(ME.sub(0).sub(1), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(5)) #uy


gdim = domain_fluid.topology.dim
gdim = 3
fdim = 1

class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        #values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
        values[0] = 1.2 * (x[1]/0.6) * (1.0 - x[1]/0.6)
        return values

class WallVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        return values


inlet_marker = 1
outlet_marker = 2
bottomedge_marker = 3
topedge_marker = 4
interface_marker = 5

# Inlet
VV, _ = ME.sub(0).collapse()
dofs_1 = fem.locate_dofs_topological( (ME.sub(0), VV), domain_fluid_facet_tags.dim, domain_fluid_facet_tags.find(1)) #vx


u_inlet = Function(VV)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)

bcu_inflow = fem.dirichletbc(u_inlet, dofs_1, ME.sub(0))
# Walls


u_nonslip = np.array((0,) * gdim, dtype=PETSc.ScalarType)

#bcu_bottomedge = fem.dirichletbc(u_inlet, fem.locate_dofs_topological(ME.sub(0), fdim, domain_fluid_facet_tags.find(bottomedge_marker)))
#bcu_topedge    = fem.dirichletbc(u_inlet, fem.locate_dofs_topological(ME.sub(0), fdim, domain_fluid_facet_tags.find(topedge_marker)))
#bcu_interface  = fem.dirichletbc(u_inlet, fem.locate_dofs_topological(ME.sub(0), fdim, domain_fluid_facet_tags.find(interface_marker)))

#bcu_bottomedge = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(ME.sub(0), fdim, domain_fluid_facet_tags.find(bottomedge_marker)))
#bcu_topedge    = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(ME.sub(0), fdim, domain_fluid_facet_tags.find(topedge_marker)))
#bcu_interface  = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(ME.sub(0), fdim, domain_fluid_facet_tags.find(interface_marker)))

#bcsu = [bcu_inflow, bcu_bottomedge, bcu_topedge, bcu_interface]

#bc1_x = fem.dirichletbc(value=1.0, dofs=dofs_1_x, V=V.sub(0))
#bc1_y = fem.dirichletbc(value=0.0, dofs=dofs_1_y, V=V.sub(1))

bc3_x = fem.dirichletbc(value=0.0, dofs=dofs_3_x, V=ME.sub(0).sub(0))
bc3_y = fem.dirichletbc(value=0.0, dofs=dofs_3_y, V=ME.sub(0).sub(1))

bc4_x = fem.dirichletbc(value=0.0, dofs=dofs_4_x, V=ME.sub(0).sub(0))
bc4_y = fem.dirichletbc(value=0.0, dofs=dofs_4_y, V=ME.sub(0).sub(1))

bc5_x = fem.dirichletbc(value=0.0, dofs=dofs_5_x, V=ME.sub(0).sub(0))
bc5_y = fem.dirichletbc(value=0.0, dofs=dofs_5_y, V=ME.sub(0).sub(1))


bcs = [bcu_inflow, bc3_x, bc3_y, bc4_x, bc4_y, bc5_x, bc5_y]



# dimension
d = len(fluid_velo)

# Kinematics
Id = Identity(d)


fluid_velo_avg = alpf*fluid_velo + (1.0-alpf)*fluid_velo_old
fluid_pres_avg = alpf*fluid_pres + (1.0-alpf)*fluid_pres_old

fluid_acce_formula = (1.0/gamm/dt)*(fluid_velo-fluid_velo_old) + ((gamm-1.0)/gamm)*fluid_acce_old

# expression for copying/storing the values
fluid_acce_expr = Expression(fluid_acce_formula, V2.element.interpolation_points())
#fluid_acce.interpolate(fluid_acce_expr)

fluid_acce_avg = alpm*fluid_acce + (1.0-alpm)*fluid_acce_old



mesh_velo_formula = (1.0/gamm/dt)*(mesh_disp_total - mesh_disp_total_old) + ((gamm-1.0)/gamm)*mesh_velo_old
# expression for copying/storing the values
mesh_velo_expr = Expression(mesh_velo_formula, V2.element.interpolation_points())
#mesh_velo.interpolate(mesh_velo_expr)


# Weak form
#Res = rho*inner(dot(2*u_old-u_old2, nabla_grad(u)), u_test)*dx_fluid + mu*inner(grad(u), grad(u_test))*dx_fluid - inner(p, div(u_test))*dx_fluid + inner(div(u), p_test)*dx_fluid

#Res =  rho*inner(fluid_acce_avg, fluid_velo_test)*dx_fluid
Res =  rho*inner(alpm*(1.0/gamm/dt)*(fluid_velo-fluid_velo_old) + (1.0-alpm/gamm)*fluid_acce_old, fluid_velo_test)*dx_fluid
Res += rho*inner(dot(fluid_velo_old, nabla_grad(fluid_velo_avg)), fluid_velo_test)*dx_fluid 
#Res += rho*inner(dot(fluid_velo_old-((1.0/gamm/dt)*(mesh_disp_total - mesh_disp_total_old) + ((gamm-1.0)/gamm)*mesh_velo_old), nabla_grad(fluid_velo_avg)), fluid_velo_test)*dx_fluid 
Res += rho*inner(dot(fluid_velo_avg, nabla_grad(fluid_velo_old)), fluid_velo_test)*dx_fluid 
Res -= rho*inner(dot(fluid_velo_old, nabla_grad(fluid_velo_old)), fluid_velo_test)*dx_fluid 

#Res =  rho*inner(dot(2*fluid_velo_old-fluid_velo_old2, nabla_grad(fluid_velo_avg)), fluid_velo_test)*dx_fluid
#Res =  rho*inner(dot(fluid_velo_old, nabla_grad(fluid_velo_avg)), fluid_velo_test)*dx_fluid
Res += mu*inner(grad(fluid_velo_avg), grad(fluid_velo_test))*dx_fluid
Res -= inner(fluid_pres_avg, div(fluid_velo_test))*dx_fluid
Res -= inner(div(fluid_velo_avg), fluid_pres_test)*dx_fluid


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




relaxFactor = 0.5 # relaxation parameter



normals_fluid = FacetNormal(domain_fluid)  # Normal pointing out of obstacle

obstacle_marker_interface_fluid = 5
ds_interface_fluid = ufl.Measure("ds", domain=domain_fluid, subdomain_data=domain_fluid_facet_tags, subdomain_id=obstacle_marker_interface_fluid)

obstacle_marker_interface_solid = 2
ds_interface_solid = ufl.Measure("ds", domain=domain_solid, subdomain_data=domain_solid_facet_tags, subdomain_id=obstacle_marker_interface_solid)


#v_test_solid = ufl.TestFunction(V2_solid)

fluid_stress = mu*grad(fluid_velo) - fluid_pres*Id

fluid_traction = fluid_stress*normals_fluid

#force_on_solid = inner( solid_disp_test, fluid_traction )  * ds_solid(2)
force_on_solid = inner(solid_disp_test, fluid_traction)  * ds_fluid(5)

#solid_force_temp = dolfinx.fem.assemble_vector(dolfinx.fem.form(L1))

solid_force_temp = dolfinx.fem.assemble_vector(dolfinx.fem.form(force_on_solid))
    
#solid_force_temp = dolfinx.fem.assemble(force_on_solid)

#L1 = v/ detJ * ufl.dx
#b1 = dolfinx.fem.assemble_vector(dolfinx.fem.form(L1))



# displacement projection
U1 = element("Lagrange", domain_fluid.basix_cell(), 1, shape=(domain_fluid.geometry.dim,))

Vs = functionspace(domain_fluid, U1)

fluid_velo_proj = Function(Vs)
fluid_velo_proj.name = "velocity"

fluid_pres_proj = Function(functionspace(domain_fluid,P1))
fluid_pres_proj.name = "pressure"


fname = "./results/Channel-thickbeam-velo-.pvd"
VTKfile_Velo = io.VTKFile(domain_fluid.comm, fname, "w")
VTKfile_Velo.write_mesh(domain_fluid)

fname = "./results/Channel-thickbeam-pres-.pvd"
VTKfile_Pres = io.VTKFile(domain_fluid.comm, fname, "w")
VTKfile_Pres.write_mesh(domain_fluid)

# function to write results to XDMF at time t
def writeResults_Velo(time_cur, timeStep):
    fluid_velo_proj.interpolate(w.sub(0))
    VTKfile_Velo.write_function(fluid_velo_proj)
    #VTKfile_Velo.write_function(mesh_velo)

# function to write results to XDMF at time t
def writeResults_Pres(time_cur, timeStep):
    fluid_pres_proj.interpolate(w.sub(1))
    VTKfile_Pres.write_function(fluid_pres_proj)


fname = "Channel-thickbeam-data.dat"
file_forces = open(fname,"w")


print("\n----------------------------\n")
print("Simulation has started")
print("\n----------------------------\n")

timeStep = 0
time_cur = 0.0

writeResults_Velo(time_cur, timeStep)
writeResults_Pres(time_cur, timeStep)
print(time_cur, 0.0, 0.0, 0.0, 0.0, 0.0, file=file_forces)


motionY_cur = 0.0
motionY_prev = 0.0


#while ( round(time_cur+dt, 6) <= time_final):
while ( round(time_cur+dt, 6) <= time_final):
    # Update current time
    time_cur += dt
    timeStep += 1

    print("\n\n Load step = ", timeStep)
    print("     Time      = ", time_cur)

    #
    # Step 1: Force predictor
    # 
    solid_force_pred = 2.0*solid_force_old - solid_force_old2

    '''    
    
    #
    # Step 2: Solve solid problem
    #
    A_solid.zeroEntries()
    assemble_matrix(A_solid, a_solid, bcs=bcs_solid)
    A_solid.assemble()
    with b_solid.localForm() as loc:
        loc.set(0)
    assemble_vector(b_solid, L_solid)
    apply_lifting(b_solid, [a_solid], [bcs_solid])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_solid, bcs_solid)
    solver_solid.solve(b_solid, solid_disp.vector)
    solid_disp.x.scatter_forward()

    # calculate velocity and acceleration
    solid_velo = (gamma_s/beta_s/dt)*(solid_disp-solid_disp_old) + (1.0-gamma_s/beta_s)*solid_velo_old + (1.0-gamma_s/2.0/beta_s)*dt*solid_acce_old
    solid_acce = (1.0/beta_s/dt/dt)*(solid_disp-solid_disp_old)  - (1.0/beta_s/dt)*solid_velo_old - (1.0/2.0/beta_s - 1.0)*solid_acce_old

    motionY_prev = motionY_cur

    motionY_cur = solid_disp

    motion_VX.value = 0.0
    motion_VY.value = solid_velo

    #motionY.value = motionY_cur - motionY_prev
    motionY.value = 0.0

    print(motionY.value)
    '''

    #
    # Step 3: Solve the mesh problem
    #
    '''
    #(num_its, converged) = solver_meshmotion.solve(mesh_disp)

    #if converged:
    #    print(f"Converged in {num_its} iterations.")
    #else:
    #    print(f"Not converged.")

    x = domain_fluid.geometry.x
    gdim = domain_fluid.geometry.dim

    #u_x = mesh_disp.compute_point_values()
    x[:,:gdim] += mesh_disp.vector[:].reshape((-1,3))

    mesh_disp_total.vector.copy(mesh_disp_total_old.vector)

    mesh_disp_total.x.array[:] += mesh_disp.x.array[:]

    #VTKfile_Velo.write_mesh(domain_fluid)
    #VTKfile_Velo.write_function(mesh_disp_total)

    mesh_velo.vector.copy(mesh_velo_old.vector)
    mesh_velo.interpolate(mesh_velo_expr)
    '''
    
    
    #
    # Step 4: Solve the fluid problem
    #

    inlet_velocity.t = time_cur
    u_inlet.interpolate(inlet_velocity)

    (num_its, converged) = solver.solve(w)

    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged.")


    # acceleration
    fluid_acce.vector.copy(fluid_acce_old.vector)
    #a_old.x.array[:] = a_temp.x.array[:]

    # compute new acceleration
    fluid_acce.interpolate(fluid_acce_expr)
    #fluid_acce = (1.0/gamm/dt)*(fluid_velo-fluid_velo_old) + ((gamm-1.0)/gamm)*fluid_acce_old


    writeResults_Velo(time_cur, timeStep)
    writeResults_Pres(time_cur, timeStep)

    # save variables
    # velocity
    #w_old2.x.array[:] = w_old.x.array
    #w_old.x.array[:] = w.x.array

    w_old.vector.copy(w_old2.vector)
    w.vector.copy(w_old.vector)

    #
    # Step 5: Correct the force
    #

    #solid_force_temp = dolfinx.fem.assemble_vector(dolfinx.fem.form(force_on_solid))
    
    solid_force_temp = dolfinx.fem.assemble(force_on_solid)

    solid_force = -relaxFactor*solid_force_temp + (1.0-beta)*solid_force_pred

    #print(time_cur, forceX, forceY, solid_disp, solid_velo, solid_acce, file=file_forces, flush=True)


    solid_force_old.vector.copy(solid_force_old2.vector)
    solid_force.vector.copy(solid_force_old.vector)

    solid_disp.vector.copy(solid_disp_old.vector)

    solid_velo.interpolate(solid_velo_expr)
    solid_velo.vector.copy(solid_velo_old.vector)

    solid_acce.interpolate(solid_acce_expr)
    solid_acce.vector.copy(solid_acce_old.vector)



VTKfile_Velo.close()
VTKfile_Pres.close()
file_forces.close()


