from numpy.linalg import solve
from fenics import *
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import os
import datetime

class DynamicTopologyOptimizer:
    def __init__(self, mesh: Mesh, V_lim: float = 1.0, q: float = 3.0, 
                 beta: float = 0.25, gamma: float = 0.5, dt: float = 1.0, 
                 T: float = 50.0, filter_radius: float = 0.05):
        """
        Initialize the dynamic topology optimization solver
        
        Args:
            mesh: FEniCS mesh
            V_lim: Volume constraint
            p: RAMP penalization parameter
            beta: Newmark-beta parameter
            gamma: Newmark-gamma parameter
            dt: Time step
            T: Total simulation time
            filter_radius: Radius for density filtering
        """
        # Initialize all attributes first to avoid __del__ errors
        self.density = None
        self.debug = False
        
        # Create results directory name file_name_datetime if it doesn't exist
        now = datetime.datetime.now()
        file_name = f"results_{now.strftime('%Y%m%d_%H%M%S')}"
        self.result_dir = os.path.join(os.getcwd(), file_name)
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.mesh = mesh
        self.V = VectorFunctionSpace(mesh, 'P', 1)
        self.S = FunctionSpace(mesh, 'P', 1)
        
        # Material parameters (Neo-Hookean)
        self.E0 = 1.0e6
        self.Emin = 1e-6
        self.nu = 0.3
        self.rho0 = 1.0
        
        # Compute Lamé parameters
        self.mu_m = self.E0/(2*(1 + self.nu))
        self.lambda_m = self.E0*self.nu/((1 + self.nu)*(1 - 2*self.nu))
        
        # Optimization parameters
        self.V_lim = V_lim
        self.q = q
        self.beta = beta
        self.gamma = gamma
        self.dt = dt  # Smaller timestep for stability
        self.T = T
        self.filter_radius = filter_radius
        
        # Initialize design variable and filtered density
        self.s = Function(self.S, name='Design Variable')
        self.s_old = Function(self.S)
        self.rho = Function(self.S, name='Filtered Density')
        s_init = np.ones(self.S.dim())*0.5
        
        # Time stepping variables
        self.n_steps = int(self.T/self.dt)
        self.t = self.dt
        
        # Initialize displacement, velocity and acceleration
        self.u = Function(self.V, name='Displacement')
        self.u_old = Function(self.V)
        self.v = Function(self.V, name='Velocity')
        self.v_old = Function(self.V)
        self.a = Function(self.V, name='Acceleration')
        self.a_old = Function(self.V)
        
        # Initialize boundary conditions
        self.boundaries = MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1)
        
        # Define boundary conditions
        self.setup_boundary_conditions()
        
        # Create measures
        self.dx = Measure('dx', domain=self.mesh)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        
        # Initialize volume
        self.vol_init =  assemble(Constant(0.5)*self.dx)
        self.s.vector().set_local(s_init)
        self.s.vector().apply('insert')
        
        # Setup density filter
        self._setup_filter()
        
        # Visualize filter after everything is set up
        self._visualize_filter_matrix()
        
        # Setup XDMF file output
        self.density_file = XDMFFile(os.path.join(self.result_dir, 'density.xdmf'))
        self.density_file.parameters["flush_output"] = True
        self.density_file.parameters["functions_share_mesh"] = True
        self.density_file.parameters["rewrite_function_mesh"] = False
        # Lists to store optimization history
        self.obj_history = []
        self.vol_history = []
        
        # Arrays to store time history for adjoint
        self.acceleration_history = []  # Store acceleration vectors
        self.tangent_history = []      # Store tangent matrices
        self.residual_forms = []       # Store residual forms
        self.M_forms = []              # Store mass matrix forms
        self.F_int_forms = []          # Store internal force forms
        self.F_ext_forms = []          # Store external force forms
        self.dM_drho_forms = []        # Store derivative of mass matrix wrt rho
        self.dF_int_drho_forms = []    # Store derivative of internal force wrt rho
        self.dF_ext_drho_forms = []    # Store derivative of external force wrt rho
        self.u_history = []            # Store displacement history
        
        # Initialize lists for storing sensitivity values
        self.mass_sens_history = []
        self.force_sens_history = []
        self.total_sens_history = []
    
    def setup_boundary_conditions(self):
        """Setup boundary conditions and force application"""
        # Create mesh function for boundaries
        self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        self.boundaries.set_all(0)
        
        # Define fixed boundary (left edge)
        class FixedBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 0, DOLFIN_EPS)
        
        # Define force boundary (right edge)
        class ForceBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 0.5, DOLFIN_EPS) and between(x[0], (0.9, 1.0))
        
        # Mark boundaries
        FIXED = 1
        FORCE = 2
        fixed = FixedBoundary()
        force = ForceBoundary()
        fixed.mark(self.boundaries, FIXED)
        force.mark(self.boundaries, FORCE)
        
        # Define time-dependent force with smaller amplitude
        force_expr = Expression(('0', 'amplitude*sin(omega*t)'),
                              amplitude=-5.0e3,  # Reduced amplitude
                              omega=np.pi/self.T,
                              t=0.0,
                              degree=1)
        
        # Create force function
        self.force = force_expr
        
        # Define Dirichlet BC (fixed boundary)
        self.bc = DirichletBC(self.V, Constant((0.0, 0.0)), self.boundaries, FIXED)
        
        # Define measures
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        self.force_marker = FORCE
        
    def _setup_filter(self):
        """Setup density filter using linear weighting based on distance"""
        # Get mesh coordinates and create KD tree for efficient neighbor search
        mesh_coordinates = self.S.tabulate_dof_coordinates()
        from scipy.spatial import cKDTree
        self.kdtree = cKDTree(mesh_coordinates)
        
        # Find neighbors within filter radius for each node
        neighbors = self.kdtree.query_ball_point(mesh_coordinates, self.filter_radius)
        
        # Compute weights for each neighbor based on distance
        weights = []
        neighbor_indices = []
        
        for i, node_neighbors in enumerate(neighbors):
            # Convert to numpy array for vectorized operations
            node_neighbors = np.array(node_neighbors)
            
            # Get distances to neighbors
            center_coord = mesh_coordinates[i]
            neighbor_coords = mesh_coordinates[node_neighbors]
            distances = np.sqrt(np.sum((neighbor_coords - center_coord)**2, axis=1))
            
            # Compute weights according to eq. (36): w_ij = (rfl - r_ij)/rfl
            # where rfl is filter radius and r_ij is distance between nodes i and j
            w = (self.filter_radius - distances) / self.filter_radius
            w = np.maximum(0, w)  # Ensure non-negative weights
            
            weights.append(w)
            neighbor_indices.append(node_neighbors)
        
        self.filter_weights = weights
        self.filter_neighbors = neighbor_indices
        
        # Debug: Visualize filter weights
        self._visualize_filter_matrix()
    
    def _visualize_filter_matrix(self):
        """Visualize the filter weights and neighbors"""
        import matplotlib.pyplot as plt
        
        # Create plots directory if it doesn't exist
        plot_dir = os.path.join(self.result_dir, 'filter_debug')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Get mesh coordinates
        coords = self.S.tabulate_dof_coordinates()
        
        # Plot filter weights for a few sample points
        plt.figure(figsize=(15, 10))
        sample_points = np.linspace(0, len(coords)-1, 5, dtype=int)
        
        for i, idx in enumerate(sample_points):
            plt.subplot(2, 3, i+1)
            
            # Get neighbors and their coordinates
            neighbors = self.filter_neighbors[idx]
            weights = self.filter_weights[idx]
            
            # Plot all mesh points in gray
            plt.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=1, alpha=0.2, label='Mesh Points')
            
            # Plot neighbors with colors based on weights
            neighbor_coords = coords[neighbors]
            scatter = plt.scatter(neighbor_coords[:, 0], neighbor_coords[:, 1], 
                                c=weights, cmap='viridis', s=30, label='Filter Weights')
            
            # Mark the center point
            plt.plot(coords[idx, 0], coords[idx, 1], 'r*', markersize=10, label='Center Node')
            
            plt.title(f'Filter Weights at Node {idx}')
            plt.colorbar(scatter, label='Weight')
            plt.axis('equal')
            if i == 0:  # Only show legend for first plot
                plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'filter_weights.png'))
        plt.close()
        
        # Plot weight distribution
        plt.figure(figsize=(10, 6))
        all_weights = np.concatenate(self.filter_weights)
        plt.hist(all_weights, bins=50, density=True)
        plt.title('Distribution of Filter Weights')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'weight_distribution.png'))
        plt.close()

    def filter_density(self):
        """Apply density filter to design variables using weighted average"""
        # Get design variables
        s = self.s.vector().get_local()
        rho_filtered = np.zeros_like(s)
        
        # Apply filter: ρₑ = (∑ᵢ w_i s_i) / (∑ᵢ w_i)
        for i in range(len(s)):
            neighbors = self.filter_neighbors[i]
            weights = self.filter_weights[i]
            
            # Compute weighted sum of densities
            rho_filtered[i] = np.sum(weights * s[neighbors]) / np.sum(weights)
        
        # Ensure bounds
        rho_filtered = np.clip(rho_filtered, 0.001, 1.0)
        
        # Update filtered density
        self.rho.vector().set_local(rho_filtered)
        self.rho.vector().apply('insert')
        
        return self.rho
        
    def get_material_properties(self):
        """Get material properties based on filtered density"""
        # Project the density to a continuous field
        rho = project(self.rho, self.S)
        
        # RAMP interpolation for Young's modulus
        E_min = self.Emin  # Small but non-zero minimum stiffness
        q = self.q  # RAMP penalization parameter
        E = E_min + (self.E0 - E_min) * rho / (1 + q*(1-rho))
        
        # Update Lamé parameters
        self.mu_m = E / (2 * (1 + self.nu))
        self.lambda_m = E * self.nu / ((1 + self.nu) * (1 - 2*self.nu))
        
        return E

    def solve_forward(self, output=False, optimization_iter=0):
        """Solve forward dynamic problem using Newmark-beta method"""
        self.u.vector().zero()
        self.v.vector().zero()
        self.a.vector().zero()
        self.u_old.vector().zero()
        self.v_old.vector().zero()
        self.a_old.vector().zero()
        # Clear previous history
        self.acceleration_history = []
        self.tangent_history = []
        self.residual_forms = []
        self.work_history = []
        self.objective_derivatives = []  # Store derivatives of objective wrt u
        self.M_forms = []              # Store mass matrix forms
        self.F_int_forms = []          # Store internal force forms
        self.F_ext_forms = []          # Store external force forms
        self.dM_drho_forms = []        # Store derivative of mass matrix wrt rho
        self.dF_int_drho_forms = []    # Store derivative of internal force wrt rho
        self.dF_ext_drho_forms = []    # Store derivative of external force wrt rho
        
        # Arrays to store norms
        self.u_norms = []
        self.v_norms = []
        self.a_norms = []
        self.tangent_norms = []
        self.residual_norms = []
        self.M_norms = []
        self.F_int_norms = []
        self.F_ext_norms = []
        self.dM_drho_norms = []
        self.dF_int_drho_norms = []
        self.dF_ext_drho_norms = []
        
        # Setup variational problem
        du = TrialFunction(self.V)  # Displacement increment
        v = TestFunction(self.V)    # Test function
        
        # Filter density and get material properties
        self.filter_density()
        self.E = self.get_material_properties()
        
        if output:
            # Setup XDMF file output at iteration i
            self.movement = XDMFFile(os.path.join(self.result_dir, f'iter_{optimization_iter}.xdmf'))
            self.movement.parameters["flush_output"] = True
            self.movement.parameters["functions_share_mesh"] = True
            self.movement.parameters["rewrite_function_mesh"] = False
            self.movement.write(self.u, 0)
            self.movement.write(self.rho, 0)
        
        # Neo-Hookean strain energy density with safeguards
        # def neo_hookean_energy(u):
        #     I = Identity(2)
        #     F = I + grad(u)
        #     C = F.T * F
        #     Ic = tr(C)
        #     J = det(F)
        #     # Add safeguard for J
        #     J = conditional(lt(J, 0.1), 0.1, J)
        #     return (self.mu_m/2)*(Ic - 3 - 2*ln(J)) + (self.lambda_m/2)*(J-1)**2
        
        # Second Piola-Kirchhoff stress with safeguards
        def PK2(u):
            I = Identity(2)
            F = I + grad(u)
            J = det(F)
            C = F.T * F
            Cinv = inv(C)
            return self.mu_m * (Identity(2) - Cinv) + self.lambda_m * (J**2 - J) * Cinv
        # First Piola-Kirchhoff stress with safeguards
        # def PK1(u):
        #     I = Identity(2)
        #     F = I + grad(u)
        #     J = det(F)
        #     # Add safeguard for J
        #     J = conditional(lt(J, 0.1), 0.1, J)
        #     Finv = inv(F)
        #     return self.mu_m * F + (self.lambda_m * ln(J) - self.mu_m) * Finv.T
        def PK1(u):
            I = Identity(2)
            F = I + grad(u)
            return F * PK2(u)

        def get_residual():
            F_int = inner(PK1(self.u), grad(v))*dx
            # Store assembled vector instead of form
            # self.F_int_forms.append(assemble(F_int))
            
            # M*u_k
            mass_term = self.rho/(self.beta*self.dt**2)
            M_form = mass_term*inner(self.u, v)*dx
            self.M_forms.append(assemble(M_form))
            
            # Add damping term
            # C_form = damping_coeff * self.rho * inner(v, (self.u - self.u_old)/self.dt)*self.dx
            
            # Mass term from equation (24)
            mass_form = self.rho * (
                self.u_old/(self.beta*self.dt**2) +
                self.v_old/(self.beta*self.dt) +
                ((1-2*self.beta)/(2*self.beta))*self.a_old
            )
            F_mass = inner(mass_form, v)*dx
            
            # External force
            F_ext = inner(self.force, v)*self.ds(2)  # Only on force boundary
            self.F_ext_forms.append(F_ext)  # Store external force form
            
            # Store derivatives wrt rho
            # dM/drho
            dM_drho = (1.0/(self.beta*self.dt**2))*inner(self.u, v)*dx
            self.dM_drho_forms.append(assemble(dM_drho))
            
            # dF_int/drho - depends on material model derivative
            E_min = self.Emin
            q = 8.0  # RAMP penalization parameter
            
            def dPK1_drho(u):
                dE_drho = (self.E0 - E_min)*(1+q)/(1 + q*(1-self.rho))**2
                # Derivative of PK1 stress wrt rho
                I = Identity(2)
                F = I + grad(u)
                J = det(F)
                C = F.T * F
                Cinv = inv(C)
                dmu_m = dE_drho/(2*(1 + self.nu))
                dlambda_m = dE_drho*self.nu/((1 + self.nu)*(1 - 2*self.nu))

                P = dmu_m * (Identity(2) - Cinv) + dlambda_m * (J**2 - J) * Cinv
                return F * P
            
            dF_int_drho = inner(dPK1_drho(self.u), grad(v))*dx
            self.dF_int_drho_forms.append(assemble(dF_int_drho))
            
            # Total residual form
            residual_form = M_form + F_int - F_ext - F_mass  # Note: F_int should be added
            
            if self.debug:
                F_ext_norm = sqrt(assemble(F_ext).norm('l2'))
                F_int_norm = sqrt(assemble(F_int).norm('l2'))
                M_norm = sqrt(assemble(M_form).norm('l2'))
                F_mass_norm = sqrt(assemble(F_mass).norm('l2')) 
                print(f"Iteration:  F_ext = {F_ext_norm:.2e}, F_int = {F_int_norm:.2e}, M = {M_norm:.2e}, F_mass = {F_mass_norm:.2e}")

            
            return residual_form
        
        # Initialize objective value
        self.objective_value = 0.0
        
        # Initialize solution vectors
        
        # Apply initial Dirichlet BCs
        # self.bc.apply(self.u.vector())
        # self.bc.apply(self.v.vector())
        # self.bc.apply(self.a.vector())
        # self.bc.apply(self.u_old.vector())
        # self.bc.apply(self.v_old.vector())
        # self.bc.apply(self.a_old.vector())
        
        # Time integration using Newmark-beta method
        for timestep in range(self.n_steps):
            # Update time
            self.t = timestep * self.dt
            
            # Update force time
            self.force.t = self.t
            
            # Newton-Raphson iteration
            max_iter = 50
            tol = 1e-5
            
            # # Predict step with smaller initial displacement
            # u_pred = self.u_old.vector().get_local() + \
            #         0.1 * self.dt * self.v_old.vector().get_local() + \
            #         0.01 * self.dt**2 * ((1-2*self.beta)/(2*self.beta))*self.a_old.vector().get_local()
            # self.u.vector().set_local(u_pred)
            # self.u.vector().apply('insert')
            residual_form = get_residual()
            residual = assemble(residual_form)
            self.bc.apply(residual)
            initial_residual_norm = residual.norm('l2') + 1e-8
            current_residual_norm = initial_residual_norm

            # Add adaptive timestep control
            dt_factor = 1.0
            if timestep > 0 and current_residual_norm > 1e3:
                dt_factor = 0.5
                self.dt *= dt_factor
                print(f"Reducing timestep to {self.dt}")
                
                # Revert to previous state
                self.u.assign(self.u_old)
                self.v.assign(self.v_old)
                self.a.assign(self.a_old)
                continue

            for k in range(max_iter):
                # Total residual form
                residual = assemble(residual_form)
                self.bc.apply(residual)
                current_residual_norm = residual.norm('l2')
                relative_residual = current_residual_norm / initial_residual_norm
                
                # Print residual norm
                if self.debug:
                    print(f"Newton-Raphson iteration {k+1}: Residual norm = {current_residual_norm:.2e}, Relative residual = {relative_residual:.2e}")
                
                # Check convergence
                if relative_residual < tol:
                    # print(f"Newton-Raphson iteration {k+1} converged after {timestep+1} time steps")
                    break
                elif k == max_iter - 1:
                    print(f"Warning: Newton-Raphson did not converge after {max_iter} iterations")
                    quit()
                
                # Compute consistent tangent matrix
                K_form = derivative(residual_form, self.u, du)
                
                # # Add inertial and stabilization terms
                # mass_term = self.rho/(self.beta*self.dt**2)
                # M_tangent = mass_term*inner(du, v)*dx
                # stab_term = 1e-6 * inner(grad(du), grad(v))*dx
                # K_form = K_form + M_tangent + stab_term
                
                # Store tangent matrix and residual form for adjoint
                K = assemble(K_form)
                self.tangent_history.append(K.copy())
                self.residual_forms.append(assemble(residual_form))
                
                # Solve linear system
                A = assemble(K_form)
                b = assemble(-residual_form)
                
                # Apply boundary conditions
                self.bc.apply(A)
                self.bc.apply(b)
                
                # Solve for displacement increment
                du_vec = Function(self.V)
                solver = LUSolver(A, "mumps")
                solver.solve(du_vec.vector(), b)
                
                # Update solution
                self.u.vector().axpy(1.0, du_vec.vector())
                
                # # Trust region approach
                # du_norm = du_vec.vector().norm('l2')
                # trust_radius = 0.1  # Initial trust region radius
                
                # if du_norm > trust_radius:
                #     scaling = trust_radius/du_norm
                #     du_vec.vector()[:] *= scaling
                #     print(f"Trust region: scaling displacement by {scaling:.2e}")
                
                # # Try update with trust region step
                # self.u.vector().axpy(1.0, du_vec.vector())
                
                # # Check if step was successful
                # new_residual = assemble(residual_form)
                # self.bc.apply(new_residual)
                # new_R_norm = new_residual.norm('l2')
                
                # if new_R_norm > current_residual_norm:
                #     # Step failed, revert and try smaller step
                #     self.u.vector().axpy(-1.0, du_vec.vector())
                #     du_vec.vector()[:] *= 0.1
                #     self.u.vector().axpy(1.0, du_vec.vector())
                #     print("Taking smaller step due to residual increase")
                
                # Apply BCs to updated solution
                self.bc.apply(self.u.vector())
            
            # Update acceleration using Newmark-beta
            a_new = (1/(self.beta*self.dt**2)) * (self.u.vector().get_local() - self.u_old.vector().get_local() - self.dt * self.v_old.vector().get_local()) - (1/(2*self.beta) - 1) * self.a_old.vector().get_local()
            self.a.vector().set_local(a_new)
            self.a.vector().apply('insert')
            self.bc.apply(self.a.vector())
            
            # Store acceleration for adjoint
            self.acceleration_history.append(self.a.vector().copy())
            
            # Update velocity using Newmark-beta
            v_new = self.gamma/(self.beta*self.dt)*(self.u.vector().get_local() - self.u_old.vector().get_local())+(1-self.gamma/self.beta)*self.v_old.vector().get_local() + \
                   self.dt * (1-self.gamma/(2*self.beta)) * self.a_old.vector().get_local() 
            self.v.vector().set_local(v_new)
            self.v.vector().apply('insert')
            self.bc.apply(self.v.vector())
            
            # Store displacement history
            self.u_history.append(self.u.vector().copy())
            
            # Calculate objective (compliance) at current timestep
            work_ext = assemble(inner(self.force, self.u)*self.ds(2))  # Only on force boundary
            
            # Store work for this timestep
            if not hasattr(self, 'work_history'):
                self.work_history = []
            self.work_history.append(work_ext)
            
            # Store derivative of objective with respect to u at this timestep
            # We'll update the scaling factor after all timesteps are done
            dwork_du = project(self.force, self.V)
            # v = TestFunction(self.V)
            # u = TrialFunction(self.V)
            # a = inner(u, v)*dx
            # L = inner(self.force, v)*ds(2)  # Only on force boundary
            F_int = inner(PK1(self.u), grad(v))*dx
            self.F_int_forms.append(assemble(F_int))
            # # Assemble system
            # A = assemble(a)
            # b = assemble(L)
            
            # # Solve system
            # solve(A, dwork_du.vector(), b)
            self.objective_derivatives.append(dwork_du.vector().copy())
            
            # Calculate objective at final timestep
            if timestep == self.n_steps - 1:
                # Initialize weights for trapezoidal rule
                weights = np.ones(self.n_steps)
                weights[0] = 0.5  # First point
                weights[-1] = 0.5  # Last point
                weights = weights * self.dt  # Scale by dt
                
                # Calculate objective using square norm of dynamic compliance
                work_squared = np.array([w**2 for w in self.work_history])
                self.objective_value = np.sqrt(np.sum(weights * work_squared))
                objective_value_no_weight = np.sqrt(np.sum( work_squared))
                
                # Update derivatives with correct scaling
                scaling_factor = 1.0 / objective_value_no_weight
                for i in range(len(self.objective_derivatives)):
                    # Scale by work at that timestep and total objective
                    self.objective_derivatives[i] *= (weights[i] * self.work_history[i] * scaling_factor)
            # print(f"Time step {timestep+1}: Objective value = {self.objective_value:.2e}")
            # Update old values
            self.u_old.assign(self.u)
            self.v_old.assign(self.v)
            self.a_old.assign(self.a)
            # self.t += self.dt
            
            if output:
                self.movement.write(self.u, timestep+1)
                self.movement.write(self.rho, timestep+1)
                
            # Store norms at each timestep
            self.u_norms.append(self.u.vector().norm('l2'))
            self.v_norms.append(self.v.vector().norm('l2'))
            self.a_norms.append(self.a.vector().norm('l2'))
            
        # After all timesteps, calculate remaining norms
        for i in range(self.n_steps):
            self.tangent_norms.append(self.tangent_history[i].norm('frobenius'))
            self.residual_norms.append((self.residual_forms[i]).norm('l2'))
            self.M_norms.append((self.M_forms[i]).norm('l2'))
            self.F_int_norms.append((self.F_int_forms[i]).norm('l2'))
            # self.F_ext_norms.append(assemble(self.F_ext_forms[i]).norm('l2'))
            self.dM_drho_norms.append((self.dM_drho_forms[i]).norm('l2'))
            self.dF_int_drho_norms.append((self.dF_int_drho_forms[i]).norm('l2'))
            # self.dF_ext_drho_norms.append(assemble(self.dF_ext_drho_forms[i]).norm('l2'))
        
        if output:
            self.movement.close()
            
            # Plot norms over time
            plot_dir = os.path.join(self.result_dir, 'norm_plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            time_points = np.arange(self.n_steps) * self.dt
            
            # Create a single figure with subplots for all variables
            plt.figure(figsize=(15, 25))  # Made taller to accommodate new plots
            
            # 1. State Variables
            # Displacement
            plt.subplot(9, 1, 1)
            plt.plot(time_points, self.u_norms, 'b-')
            plt.xlabel('Time')
            plt.ylabel('Norm')
            plt.title('Displacement History')
            plt.grid(True)
            
            # Velocity
            plt.subplot(9, 1, 2)
            plt.plot(time_points, self.v_norms, 'r-')
            plt.xlabel('Time')
            plt.ylabel('Norm')
            plt.title('Velocity History')
            plt.grid(True)
            
            # Acceleration
            plt.subplot(9, 1, 3)
            plt.plot(time_points, self.a_norms, 'g-')
            plt.xlabel('Time')
            plt.ylabel('Norm')
            plt.title('Acceleration History')
            plt.grid(True)
            
            # 2. System and Force Terms
            # Tangent Matrix
            plt.subplot(9, 1, 4)
            plt.plot(time_points, self.tangent_norms, 'b-')
            plt.xlabel('Time')
            plt.ylabel('Norm')
            plt.title('Tangent Matrix History')
            plt.grid(True)
            
            # Mass Matrix
            plt.subplot(9, 1, 5)
            plt.plot(time_points, self.M_norms, 'r-')
            plt.xlabel('Time')
            plt.ylabel('Norm')
            plt.title('Mass Matrix History')
            plt.grid(True)
            
            # Internal Force
            plt.subplot(9, 1, 6)
            plt.plot(time_points, self.F_int_norms, 'g-')
            plt.xlabel('Time')
            plt.ylabel('Norm')
            plt.title('Internal Force History')
            plt.grid(True)
            
            # 3. Sensitivity Terms
            # Mass Matrix Sensitivity
            plt.subplot(9, 1, 7)
            plt.plot(time_points, self.dM_drho_norms, 'b-')
            plt.xlabel('Time')
            plt.ylabel('Norm')
            plt.title('Mass Matrix Sensitivity')
            plt.grid(True)
            
            # Internal Force Sensitivity
            plt.subplot(9, 1, 8)
            plt.plot(time_points, self.dF_int_drho_norms, 'g-')
            plt.xlabel('Time')
            plt.ylabel('Norm')
            plt.title('Internal Force Sensitivity')
            plt.grid(True)
            
            # Work History
            plt.subplot(9, 1, 9)
            plt.plot(time_points, self.work_history, 'k-')
            plt.xlabel('Time')
            plt.ylabel('Work')
            plt.title('Work History')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'all_variables_iter_{optimization_iter}.png'))
            plt.close()
        
        # Calculate adjoint variables
        self.calculate_adjoints()
            
        return self.objective_value
    
    def get_sensitivity(self):
        """Calculate sensitivity df/drho using adjoint method"""
        # Get number of density variables (half of displacement variables)
        n_dens = int(len(self.s.vector().get_local()))
        total_sensitivity = np.zeros(2*n_dens)  # Initialize in displacement space
        
        # Define test function for force sensitivity calculation
        v = TestFunction(self.V)
        
        def dPK1_drho(u):
            """Calculate derivative of PK1 stress with respect to density"""
            # Material parameters
            E_min = self.Emin
            q = self.q  # RAMP penalization parameter
            
            # Derivative of Young's modulus wrt density using RAMP
            dE_drho = (self.E0 - E_min)*(1+q)/(1 + q*(1-self.rho))**2
            
            # Derivative of Lamé parameters
            dmu_m = dE_drho/(2*(1 + self.nu))
            dlambda_m = dE_drho*self.nu/((1 + self.nu)*(1 - 2*self.nu))
            
            # Kinematics
            I = Identity(2)
            F = I + grad(u)
            J = det(F)
            C = F.T*F
            Cinv = inv(C)
            
            # Derivative of PK1 with respect to density
            P = dmu_m * (F - inv(F.T)) + dlambda_m * ln(J) * inv(F.T)
            return P
        
        # Rest of the function remains the same
        for i in range(self.n_steps):
            # Current solution
            a = Function(self.V)
            a.vector()[:] = self.acceleration_history[i].get_local()
            
            lambda_adj = Function(self.V)
            lambda_adj.vector()[:] = self.lambda_adj[i].vector().get_local()
            
            # Mass matrix sensitivity: dM/drho * a * lambda
            dM = (self.dM_drho_forms[i])
            mass_sens = dM.get_local() * a.vector().get_local() * lambda_adj.vector().get_local()  # Positive because M is positive in residual
            
            # Force sensitivity: dF/drho * lambda
            # Get current displacement for this timestep
            u = Function(self.V)
            u.vector().set_local(self.u_history[i].get_local())
            
            # Calculate force sensitivity form
            dF_form = inner(dPK1_drho(u), grad(v))*dx
            dF = assemble(dF_form)
            force_sens = dF.get_local() * lambda_adj.vector().get_local()  # Positive because F_int is positive in residual
            
            # Store sensitivities for plotting
            self.mass_sens_history.append(mass_sens)
            self.force_sens_history.append(force_sens)
            self.total_sens_history.append(mass_sens + force_sens)
            
            # Add sensitivities
            total_sensitivity += mass_sens + force_sens
        
        # Convert sensitivity from displacement space to density space
        # Each density point corresponds to x,y components in displacement
        sens_density = np.zeros(n_dens)
        for i in range(n_dens):
            # Sum x,y components for each node
            sens_density[i] = total_sensitivity[2*i] + total_sensitivity[2*i + 1]
        
        plt.figure()
        plt.plot(sens_density)
        plt.savefig(self.result_dir + '/df_drho.png')
        plt.close()
        
        
        # Apply filter weights
        # filtered_sens = self.filter_sensitivity(sens_density)
        
        # Create final sensitivity function in density space
        # sensitivity_final = Function(self.S)
        # sensitivity_final.vector()[:] = filtered_sens
            
        #save graph of sensitivity
        # plt.figure()
        # plt.plot(sensitivity_final.vector().get_local())
        # plt.savefig(self.result_dir + '/sensitivity.png')
        # plt.close()

        # Plot sensitivity history
        plt.figure(figsize=(15, 10))
        sens_time_points = np.arange(len(self.mass_sens_history)) * self.dt
        plt.subplot(2, 1, 1)
        mass_sens_norms = [np.linalg.norm(sens) for sens in self.mass_sens_history]
        force_sens_norms = [np.linalg.norm(sens) for sens in self.force_sens_history]
        plt.plot(sens_time_points, mass_sens_norms, 'b-', label='Mass')
        plt.plot(sens_time_points, force_sens_norms, 'r-', label='Force')
        plt.xlabel('Time')
        plt.ylabel('Norm')
        plt.title('Mass and Force Sensitivity History')
        plt.legend()
        plt.grid(True)
        
        # Total Sensitivity
        plt.subplot(2, 1, 2)
        total_sens_norms = [np.linalg.norm(sens) for sens in self.total_sens_history]
        plt.plot(sens_time_points, total_sens_norms, 'k-')
        plt.xlabel('Time')
        plt.ylabel('Norm')
        plt.title('Total Sensitivity History')
        plt.grid(True)
        plt.savefig(self.result_dir + '/total_sensitivity.png')
        plt.close()

        return sens_density
    
    def filter_sensitivity(self, sens_array):
        """Filter sensitivity according to equation (35)"""
        # Get mesh coordinates
        coords = self.S.tabulate_dof_coordinates()
        n_nodes = len(coords)
        
        # Initialize filtered sensitivity
        filtered_sens = np.zeros_like(sens_array)
        
        # For each node
        for i in range(n_nodes):
            # Get current node coordinates
            xi = coords[i]
            
            # Calculate distances to all other nodes
            dists = np.linalg.norm(coords - xi, axis=1)
            
            # Calculate weights
            weights = np.maximum(0, self.filter_radius - dists)
            weight_sum = np.sum(weights)
            
            if weight_sum > 0:
                # Apply filter: df/ds_i = sum_j (df/drho_j * w_ji / sum_k w_jk)
                filtered_sens[i] = np.sum(sens_array * weights) / weight_sum
        
        return filtered_sens
    
    def calculate_adjoints(self):
        """Calculate adjoint variables μ, η, and λ for sensitivity analysis"""
        # Initialize adjoint variables for each timestep
        self.mu = [Function(self.V) for _ in range(self.n_steps)]
        self.eta = [Function(self.V) for _ in range(self.n_steps)]
        self.lambda_adj = [Function(self.V) for _ in range(self.n_steps)]
        
        # Last timestep adjoints are zero
        self.mu[-1].vector().zero()
        self.eta[-1].vector().zero()
        self.lambda_adj[-1].vector().zero()
        
        # Get mass matrix M and dt
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        
        # Assemble mass matrix (constant throughout simulation)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        M = assemble(self.rho * inner(u, v)*dx)  # Basic mass matrix without coefficients
        
        # Calculate adjoints backwards in time
        for i in range(self.n_steps-2, -1, -1):  # n-2 to 0
            df_du = self.objective_derivatives[i]
            
            # Setup and solve system for μᵢ (equation 68)
            # μᵢ = ∂f/∂aᵢ + (∂R_i+1/∂u_i)λᵢ₊₁ + (∂H_i+1/∂a_i)muᵢ₊1 + (∂L_i+1/∂a_i)ηᵢ₊₁
            temp_vec = Function(self.V).vector()
            
            # Term with M * lambda
            M.mult(self.lambda_adj[i+1].vector(), temp_vec)
            self.mu[i].vector().axpy(-(1/(2*beta)-1), temp_vec)
            
            # Term with mu
            self.mu[i].vector().axpy(-(1/(2*beta)-1), self.mu[i+1].vector())
            
            # Term with eta
            self.mu[i].vector().axpy((1-gamma/(2*beta))*dt, self.eta[i+1].vector())
            
            # Setup and solve system for ηᵢ (equation 69)
            # ηᵢ = ∂f/∂vᵢ + (∂R/∂v)ᵢᵀλᵢ₊₁ + (∂H/∂v)ᵢᵀλᵢ₊₂ + (∂L/∂v)ᵢᵀηᵢ₊₁
            temp_vec.zero()
            M.mult(self.lambda_adj[i+1].vector(), temp_vec)
            self.eta[i].vector().axpy(-gamma/(beta*dt), temp_vec)
            
            self.eta[i].vector().axpy(-1.0/(beta*dt), self.mu[i+1].vector())
            self.eta[i].vector().axpy(1.0 - gamma/beta, self.eta[i+1].vector())
            
            # Setup and solve system for λᵢ (equation 70)
            # dR/duᵢ λᵢ = -∂f/∂uᵢ - (∂R/∂a)ᵢᵀλᵢ₊₁ - (∂H/∂a)ᵢᵀλᵢ₊₂ - (∂L/∂a)ᵢᵀηᵢ₊₁
            rhs_lambda = df_du.copy()
            rhs_lambda *= -1.0  # -∂f/∂uᵢ
            
            # Term with M * lambda
            temp_vec.zero()
            M.mult(self.lambda_adj[i+1].vector(), temp_vec)
            rhs_lambda.axpy(1.0/(beta*dt*dt), temp_vec)
            
            # Terms with mu difference
            rhs_lambda.axpy(-1.0/(beta*dt*dt), self.mu[i].vector())
            rhs_lambda.axpy(1.0/(beta*dt*dt), self.mu[i+1].vector())
            
            # Terms with eta difference
            rhs_lambda.axpy(-gamma/(beta*dt), self.eta[i].vector())
            rhs_lambda.axpy(gamma/(beta*dt), self.eta[i+1].vector())
            
            # Solve system
            # We should use the tangent matrix from the forward solve since it contains both mass and stiffness
            solver = LUSolver(self.tangent_history[i], "mumps")
            solver.solve(self.lambda_adj[i].vector(), rhs_lambda)
            
            # Apply boundary conditions
            self.bc.apply(self.lambda_adj[i].vector())
        
        # Plot all adjoint norms over time
        mu_norm = np.zeros(self.n_steps)
        eta_norm = np.zeros(self.n_steps)
        lambda_norm = np.zeros(self.n_steps)

        for i in range(self.n_steps):
            mu_norm[i] = np.linalg.norm(self.mu[i].vector().get_local())
            eta_norm[i] = np.linalg.norm(self.eta[i].vector().get_local())
            lambda_norm[i] = np.linalg.norm(self.lambda_adj[i].vector().get_local())

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(self.n_steps), mu_norm)
        plt.xlabel("Time step")
        plt.ylabel("μ norm")
        plt.title("Adjoint μ")

        plt.subplot(3, 1, 2)
        plt.plot(np.arange(self.n_steps), eta_norm)
        plt.xlabel("Time step")
        plt.ylabel("η norm")
        plt.title("Adjoint η")

        plt.subplot(3, 1, 3)
        plt.plot(np.arange(self.n_steps), lambda_norm)
        plt.xlabel("Time step")
        plt.ylabel("λ norm")
        plt.title("Adjoint λ")
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/adjoint_norm.png')
        plt.close()
    
    def calculate_sensitivity(self):
        """Calculate sensitivity of objective function"""
        # Get current density distribution
        rho = self.rho
        
        # Test function for sensitivity calculation
        v = TestFunction(self.S)
        
        # Material interpolation derivative
        E_min = self.Emin
        q = 8.0  # RAMP penalization parameter
        dE_drho = (self.E0 - E_min) *(1+rho)/ (1 + q*(1-rho))**2
        
        # Current displacement
        u = self.u
        
        # Sensitivity of internal energy
        F = Identity(2) + grad(u)
        C = F.T * F
        E = tr(C)
        J = det(F)
        
        # Neo-Hookean strain energy density derivative
        dW_drho = (dE_drho/2)*(E - 3 - 2*ln(J)) + (dE_drho*self.nu/2)*(J-1)**2
        
        # Create sensitivity form
        L_form = v * dW_drho * self.dx
        
        # Compute sensitivity vector
        b = assemble(L_form)
        
        # Apply chain rule through the filter
        dc_filtered = Function(self.S)
        dc_filtered.vector()[:] = self.H @ (b.get_local() / self.Hs)
        
        return dc_filtered.vector().get_local()
    
    def update_design(self, sensitivity):
        """Update design variables using OC method"""
        # Get current design and sensitivity
        s_old = self.s.vector().get_local()
        
        # Convert sensitivity to numpy array if it's a Function
        if isinstance(sensitivity, Function):
            dc = sensitivity.vector().get_local()
        else:
            dc = sensitivity
            
        # Parameters for bisection
        move = 0.1
        l1 = 1e-8
        l2 = 1e9
        eps = 1e-6  # Small number to avoid division by zero
        
        # Volume constraint
        vol = assemble(self.rho*dx)/self.vol_init
        
        while (l2-l1)/(l1+l2+eps) > 1e-3:
            lmid = 0.5*(l2+l1)
            s_new = np.zeros(s_old.shape)
            
            # Optimality criteria
            B = -dc/(lmid + eps)  # Add eps to avoid division by zero
            
            # Handle negative/zero/NaN values
            B = np.nan_to_num(B, nan=0.0, posinf=1e9, neginf=-1e9)
            
            # Update design variables with bounds
            s_new = np.maximum(0.001, np.maximum(s_old - move, 
                    np.minimum(1.0, np.minimum(s_old + move, 
                    s_old*np.sqrt(np.maximum(eps, B))))))  # Ensure positive value before sqrt
            
            # Filter design variables
            self.s.vector().set_local(s_new)
            self.s.vector().apply('insert')
            self.filter_density()
            
            # Update volume fraction
            vol = assemble(self.rho*dx)/self.vol_init
            
            # Update Lagrange multiplier
            if vol > self.V_lim:
                l1 = lmid
            else:
                l2 = lmid
        
        # Set final design
        # Handle any remaining NaN values
        s_new = np.nan_to_num(s_new, nan=0.0)
        self.s.vector().set_local(s_new)
        self.s.vector().apply('insert')
    
    def optimize(self, max_iter: int = 1000, tol: float = 1e-4):
        """Run optimization loop"""
        for optimization_iter in range(max_iter):
            # Store old design
            self.s_old.assign(self.s)
            
            # Filter density
            self.filter_density()
            
            # Solve forward problem and get objective
            if optimization_iter % 5 == 0:
                self.output = True
            else:
                self.output = False
            obj = self.solve_forward(self.output, optimization_iter)
            
            # Calculate objective
            # obj = self.calculate_objective()
            vol_frac = assemble(self.rho*dx)/self.vol_init
            
            # Store history
            self.obj_history.append(obj)
            self.vol_history.append(vol_frac)
            
            # Get sensitivity
            sens = self.get_sensitivity()
            
            # Convert sensitivity array to FEniCS Function
            sens_func = Function(self.S)
            sens_func.vector().set_local(sens)
            sens_func.vector().apply('insert')
            
            # Filter sensitivities using the same filter as density
            sens_array = sens_func.vector().get_local()
            filtered_sens = np.zeros_like(sens_array)
            for i in range(len(sens_array)):
                neighbors = self.filter_neighbors[i]
                weights = self.filter_weights[i]
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    filtered_sens[i] = np.sum(sens_array[neighbors] * weights) / weight_sum
            sens_func.vector().set_local(filtered_sens)
            sens_func.vector().apply('insert')
            
            # Update design using gradient descent with adaptive step size
            sens_array = sens_func.vector().get_local()
            sens_norm = np.linalg.norm(sens_array)
            if sens_norm > 0:
                # Adaptive step size based on sensitivity magnitude
                step_size = min(1.0, 2.0/sens_norm)
                self.s.vector()[:] -= step_size * sens_array  # Minimize objective
            
            # Clip design variables
            self.s.vector()[:] = np.clip(self.s.vector()[:], 0.001, 1.0)
            
            # Update filtered density based on design variables
            self.filter_density()
            
            # Enforce minimum density at force boundary and specified region
            force_boundary_dofs = set()  # Use set to avoid duplicates
            
            # Create function to mark force boundary vertices
            force_boundary = MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1, 0)
            for facet in facets(self.mesh):
                if self.boundaries[facet] == 2:  # Force boundary
                    force_boundary[facet] = 1
            
            # Get density function space coordinates
            S_dofmap = self.S.dofmap()
            S_coordinates = self.S.tabulate_dof_coordinates()
            
            # Find density DOFs on force boundary
            for cell in cells(self.mesh):
                for facet in facets(cell):
                    if force_boundary[facet] == 1:
                        # Get density DOFs for this cell
                        cell_dofs = S_dofmap.cell_dofs(cell.index())
                        for dof in cell_dofs:
                            x, y = S_coordinates[dof]
                            if x > 0.9 and y > 0.4:
                                force_boundary_dofs.add(dof)
            
            force_boundary_dofs = list(force_boundary_dofs)
            # print("Force boundary DOFs:", force_boundary_dofs)
            
            # Set minimum density of 1.0 at force boundary and specified region
            rho_array = self.rho.vector().get_local()
            rho_array[force_boundary_dofs] = 1.0  # Force exactly 1.0
            # Global density bounds
            rho_array = np.clip(rho_array, 0.001, 1.0)
            self.rho.vector().set_local(rho_array)
            self.rho.vector().apply('insert')
            
            # Save results to XDMF and plot progress
            if optimization_iter % 5 == 0 or optimization_iter == max_iter-1:  # Save every 5 iterations and final iteration
                self.save_results(optimization_iter)
                self.plot_progress()
            
            # Print iteration info
            design_change = sqrt(assemble((self.s - self.s_old)**2*dx))
            obj_change = float('inf')
            if len(self.obj_history) > 2:
                obj_change = abs(obj - self.obj_history[-2])/max(abs(obj), 1e-10)
            
            print(f"Iteration {optimization_iter}: obj = {obj:.3e}, vol = {vol_frac:.3f}, design change = {design_change:.3e}, relative obj change = {obj_change:.3e}")
            self.density_file.write(self.rho, optimization_iter)
            # Check convergence
            if design_change < tol and obj_change < tol:
                print("Convergence achieved!")
                self.save_results(optimization_iter)
                self.plot_progress()
                break
    
    def save_results(self, iteration: int):
        """Save current results to XDMF files"""
        # Create results directory if it doesn't exist
        plot_dir = self.result_dir + '/plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)


            
        # Save density plot
        plt.figure(figsize=(10, 8))
        c = plot(self.rho)
        plt.colorbar(c, label='Density')
        plt.title(f'Density Distribution - Iteration {iteration}')
        plt.savefig(f'{plot_dir}/density_{iteration:02d}.png')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        c = plot(self.rho)
        plt.colorbar(c, label='Density')
        plt.title(f'Density Distribution - Iteration {iteration}')
        plt.savefig(f'{plot_dir}/current_density.png')
        plt.close()
        # Save displacement plot
        # plt.figure(figsize=(10, 8))
        # c = plot(self.u)
        # plt.colorbar(c, label='Displacement Magnitude')
        # plt.title(f'Displacement Field - Iteration {iteration}')
        # plt.savefig(f'{plot_dir}/displacement_{iteration}.png')    
        # plt.close()
        
        # Save convergence history
        if len(self.obj_history) > 0:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.obj_history)
            plt.xlabel('Iteration')
            plt.ylabel('Objective')
            plt.title('Objective History')
            
            plt.subplot(1, 2, 2)
            plt.plot(self.vol_history)
            plt.xlabel('Iteration')
            plt.ylabel('Volume Fraction')
            plt.title('Volume Fraction History')
            
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/convergence.png')
            plt.close()
            

        # Save to XDMF for ParaView visualization
        self.density_file.write(self.rho, iteration)
    
    def plot_progress(self):
        """Plot optimization progress"""
        # Create plots directory if it doesn't exist
        plot_dir = os.path.join(self.result_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot convergence history
        plt.figure(figsize=(10, 6))
        
        # Plot objective on left y-axis
        ax1 = plt.gca()
        line1 = ax1.plot(self.obj_history, 'b-', label='Objective')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot volume on right y-axis
        ax2 = ax1.twinx()
        line2 = ax2.plot(self.vol_history, 'r--', label='Volume')
        ax2.set_ylabel('Volume Fraction', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title('Convergence History')
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/plots/progress.png')
        plt.close()
    
    def __del__(self):
        """Clean up XDMF files"""
        if hasattr(self, 'density_file') and self.density_file is not None:
            try:
                self.density_file.close()
            except:
                pass

# Example usage
if __name__ == "__main__":
    # Create mesh (dimensions from the image: 1.0m x 0.5m)
    mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 0.5), 50, 50)
    
    # Initialize optimizer
    optimizer = DynamicTopologyOptimizer(mesh)
    
    # Run optimization
    optimizer.optimize()
