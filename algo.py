import numpy as np
from sympy import symbols, Eq, solve
import tensorflow as tf
from tqdm import tqdm
import os
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
import sys
import time



def load_and_update_model(mass):
    global model  

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    MODEL_PATH = f"mismatch_allmodes_{mass}"

    # Load the model and update the global variable
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def convert_to_lambda(q, spin1, spin2, theta1, theta2):
    
    eta = q / (1 + q)**2
    phi1, phi2 = np.random.uniform(0, 2 * np.pi, 2)
    a1z = spin1 * np.cos(theta1)
    a2z = spin2 * np.cos(theta2)
    a1x = spin1 * np.sin(theta1) * np.cos(phi1)
    a1y = spin1 * np.sin(theta1) * np.sin(phi1)
    a2x = spin2 * np.sin(theta2) * np.cos(phi2)
    a2y = spin2 * np.sin(theta2) * np.sin(phi2)

    return [eta, a1x, a1y, a1z, a2x, a2y, a2z]



class BBH:
    def __init__(self, lam=None, precession=True, noalign=False, inplane = False):
        """
        Binary Black Hole (BBH) initialization.

        :param lam: List of parameters [eta, a1x, a1y, a1z, a2x, a2y, a2z].
        :param precession: If False, sets in-plane spins (x, y) to zero for both.
        :param noalign: If True, primary has only in-plane spin (no z component),
                        while secondary has only aligned (z) spin.
        """
        self.precession = precession
        self.noalign = noalign
        self.inplane = inplane

        if lam is None:
            self.random()
        else:
            self.lam = lam
            self.initialize_from_lambda()

        # Ensure z1 and z2 are always set
        self.z1 = self.a1[2]
        self.z2 = self.a2[2]

        # Compute derived quantities
        self.find_chi_p()
        self.find_chi_eff()

    def random(self):
        """Generates a random BBH system with precessing or aligned spins."""
        self.initialize_mass_ratio()
        self.construct_spin_vectors()

        # Compute theta, phi, and in-plane components explicitly
        self.inplane1 = find_spin_mag(self.a1[:2])
        self.inplane2 = find_spin_mag(self.a2[:2])
        self.theta1 = find_theta(self.z1, self.spin1)
        self.theta2 = find_theta(self.z2, self.spin2)
        self.phi1 = find_phi(self.x1, self.y1)
        self.phi2 = find_phi(self.x2, self.y2)

        self.construct_features_array()

    def initialize_from_lambda(self):
        """Initializes the BBH from a given lambda parameter list."""
        assert len(self.lam) == 7, "The 'lam' list must contain exactly 7 entries."

        self.eta = self.lam[0]
        self.q_from_eta()
        assert 1 <= self.q <= 6, "mass ratio out of bounds"

        self.a1 = self.lam[1:4]
        self.a2 = self.lam[4:7]

        if not self.precession:
            # No precession: aligned spins only (z-axis)
            self.a1[0] = self.a1[1] = 0  
            self.a2[0] = self.a2[1] = 0  

        if self.inplane:
            # No aligned component for primary: only x, y spin
            self.a1[2] = 0  # Primary has no aligned (z) spin
            self.a2[0] = self.a2[1] = self.a2[2] =0  # Secondary remains aligned (no in-plane)

        # Set all parameters explicitly
        self.x1, self.y1, self.z1 = self.a1
        self.x2, self.y2, self.z2 = self.a2
        self.spin1 = find_spin_mag(self.a1)
        self.spin2 = find_spin_mag(self.a2)
        self.inplane1 = find_spin_mag(self.a1[:2])
        self.inplane2 = find_spin_mag(self.a2[:2])
        self.theta1 = find_theta(self.z1, self.spin1)
        self.theta2 = find_theta(self.z2, self.spin2)
        self.phi1 = find_phi(self.x1, self.y1)
        self.phi2 = find_phi(self.x2, self.y2)

    def initialize_mass_ratio(self):
        """Randomly generates mass ratio q and symmetric mass ratio eta."""
        self.q = np.random.uniform(1, 6)
        self.eta = self.q / (self.q + 1) ** 2

    def construct_spin_vectors(self):
        """Constructs spin vectors, enforcing aligned-only or no-aligned conditions."""
        if self.precession and not self.noalign and not self.inplane:
            # Normal precessing case (random 3D spins)
            self.a1, self.spin1, self.theta1, self.phi1 = self.random_spin_vector()
            self.a2, self.spin2, self.theta2, self.phi2 = self.random_spin_vector()

        elif not self.precession:
            # No precession: only aligned spins (z-component)
            self.a1 = [0, 0, np.random.uniform(-1, 1)]  # Primary spin only along z-axis
            self.a2 = [0, 0, np.random.uniform(-1, 1)]  # Secondary spin only along z-axis

            # Ensure spin magnitudes are set
            self.spin1 = abs(self.a1[2])  
            self.spin2 = abs(self.a2[2])

            # Set angles correctly (theta is either 0 or π)
            self.theta1 = 0 if self.a1[2] >= 0 else np.pi
            self.theta2 = 0 if self.a2[2] >= 0 else np.pi
            self.phi1 = 0  # No in-plane spin means phi is undefined
            self.phi2 = 0

        elif self.noalign:
            # Primary: no aligned spin, only x, y components
            self.a1, self.spin1, self.theta1, self.phi1 = self.random_inplane_spin_vector()

            # Secondary: only aligned spin
            self.a2 = [0, 0, 0]
            self.spin2 = 0
            self.theta2 = 0 
            self.phi2 = 0
        
        elif self.inplane:
            self.a1, self.spin1, self.theta1, self.phi1 = self.random_inplane_spin_vector()
            self.a2, self.spin2, self.theta2, self.phi2 = self.random_inplane_spin_vector()


        # Ensure all parameters are set
        self.x1, self.y1, self.z1 = self.a1
        self.x2, self.y2, self.z2 = self.a2
        self.inplane1 = find_spin_mag(self.a1[:2])
        self.inplane2 = find_spin_mag(self.a2[:2])

    def construct_features_array(self):
        """Constructs a parameter array for the BBH system."""
        self.lam = [self.eta] + self.a1 + self.a2

    def find_chi_p(self):
        frac = (4 + 3 * self.q) / (4 * self.q**2 + 3 * self.q)
        self.chi_p = max(self.inplane1, frac * self.inplane2)
        assert 0 <= self.chi_p <= 1, "chi_p out of bounds"

    def find_chi_eff(self):
        self.chi_eff = (self.q * self.a1[2] + self.a2[2]) / (self.q + 1)
        assert -1 <= self.chi_eff <= 1, "chi_eff out of bounds"

    def q_from_eta(self):
        q = symbols('q')
        equation = Eq(self.eta, q / (q + 1)**2)
        solutions = solve(equation, q)
        valid_solutions = [sol for sol in solutions if sol >= 1]
        self.q = float(valid_solutions[0])

    def random_spin_vector(self):
        """Generates a random 3D spin vector."""
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        r = np.random.uniform(0, 1)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return [x, y, z], r, theta, phi

    def random_inplane_spin_vector(self):
        """Generates a random 2D in-plane spin vector (no z component)."""
        phi = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 1)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return [x, y, 0], r, np.pi / 2, phi  # Theta is π/2 (pure in-plane)


def find_spin_mag(vec):
    mag = np.linalg.norm(vec)
    return mag

def find_theta(z, spin):
    assert 0<= np.abs(z)<=spin, "Aligned spin mag should always be between 0 and spin magnitude"
    if spin == 0:
        theta = 0
    else:
        theta = np.arccos(z/spin)
    # if theta < 0:
    #     theta = np.pi + theta
    return theta

def find_phi(x, y):
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2*np.pi
    return phi

    

def find_inplane(theta,spin):
    inplane = spin * np.sin(theta)
    return inplane


class ParameterSpace:
    PARAM_ATTRIBUTES = {
        "eta": 'eta',
        "q": 'q',
        "chieff": 'chi_eff',
        "spin1": 'spin1',
        "spin2": 'spin2',
        "theta1": 'theta1',
        "theta2": 'theta2',
        "inplane1":'inplane1',
        "inplane2":'inplane2',
        "mismatch": 'mismatch',
        "chip": 'chi_p',
        "phi1":'phi1',
        "phi2":'phi2',
        "z1": 'z1',
        "z2": 'z2',
        "x1": 'x1',
        "y1": 'y1',
        "x2": 'x2',
        "y2": 'y2'
    }

    PARAM_STRINGS = {
        "eta": "$\eta$",
        "q": "$q$",
        "chieff": "$\chi_{\mathrm{eff}}$",
        "spin1": "$a_{1}$",
        "spin2": "$a_{2}$",
        "theta1": "$\\theta_1$",
        "theta2": "$\\theta_2$",
        "phi1": "$\\phi_1$",
        "phi2": "$\\phi_2$",
        "inplane1":"$\chi_{1\perp}",
        "inplane2":"$\chi_{2\perp}",
        "mismatch": "$\mathcal{MM}$",
        "chip": "$\chi_{p}$",
        "z1": '$a_{1z}$',
        "z2": '$a_{2z}$'
    }

    PARAM_BOUNDS = {
        "eta": (6/49, 0.25),
        "q": (1, 6),
        "chieff": (-1, 1),
        "spin1": (0, 1),
        "spin2": (0, 1),
        "theta1": (0, np.pi),
        "theta2": (0, np.pi),
        "phi1": (0, 2*np.pi),
        "phi2": (0, 2*np.pi),
        "mismatch": (0, 1),
        "chip": (0, 1),
        "inplane1":(0, 1),
        "inplane2":(0, 1),
        "z1": (-1,1),
        "z2": (-1,1),
        "x1": (-1,1),
        "y1": (-1,1),
        "x2": (-1,1),
        "y2": (-1,1)
    }


    def __init__(self, sample=10000, lam0 = None, inj = False, data = None, start = None, 
                 precession = True, noalign = False, inplane = False):
        
        self.lam0 = lam0
        self.sample = sample
        self.inj = inj
        self.precession = precession
        self.noalign = noalign
        self.inplane = inplane
        self.bbh_instances = []  # List to store BBH instances
                
        if self.lam0 is not None:
            bbh0 = BBH(lam = lam0)
            self.bbh_instances.append(bbh0)
            features = self.lam0 + self.lam0
            sample = sample - 1
            
        if self.inj: 
            for _ in tqdm(range(sample)):
                self.data = data
                self.start = start
                perturbed_lam = self.perturb_lam()
                bbh = BBH(lam=perturbed_lam)  
                self.bbh_instances.append(bbh)
                          
        elif not self.precession:
            for _ in tqdm(range(0, sample)):
                bbh = BBH(precession = False)
                self.bbh_instances.append(bbh)
        elif self.noalign:
            for _ in tqdm(range(0, sample)):
                bbh = BBH(noalign = True)
                self.bbh_instances.append(bbh)
        elif self.inplane:
            for _ in tqdm(range(0, sample)):
                bbh = BBH(inplane = True)
                self.bbh_instances.append(bbh)
        else:
            for _ in tqdm(range(0, sample)):
                bbh = BBH()
                self.bbh_instances.append(bbh)
        
     
        # Extract data for easy access
        self.eta = np.array([bbh.eta for bbh in self.bbh_instances])
        self.q = np.array([bbh.q for bbh in self.bbh_instances])
        self.chi_eff = np.array([bbh.chi_eff for bbh in self.bbh_instances])
        self.spin1 = np.array([bbh.spin1 for bbh in self.bbh_instances])
        self.spin2 = np.array([bbh.spin2 for bbh in self.bbh_instances])
        self.inplane1 = np.array([bbh.inplane1 for bbh in self.bbh_instances])
        self.inplane2 = np.array([bbh.inplane2 for bbh in self.bbh_instances])
        self.z1 = np.array([bbh.z1 for bbh in self.bbh_instances])
        self.z2 = np.array([bbh.z2 for bbh in self.bbh_instances])
        self.chi_p = np.array([bbh.chi_p for bbh in self.bbh_instances])
        self.theta1 = np.array([bbh.theta1 for bbh in self.bbh_instances])
        self.theta2 = np.array([bbh.theta2 for bbh in self.bbh_instances])
        self.phi1 = np.array([bbh.phi1 for bbh in self.bbh_instances])
        self.phi2 = np.array([bbh.phi2 for bbh in self.bbh_instances])
        self.x1 = np.array([bbh.x1 for bbh in self.bbh_instances])
        self.x2 = np.array([bbh.x2 for bbh in self.bbh_instances])
        self.y1 = np.array([bbh.y1 for bbh in self.bbh_instances])
        self.y2 = np.array([bbh.y2 for bbh in self.bbh_instances])
        
 
        self.lams = np.array([bbh.lam for bbh in self.bbh_instances])
        
        
        features_list = []
        
        if self.lam0 is not None:
            for lam in self.lams:
                f = np.concatenate((lam0, lam))
                features_list.append(f)
            
            features = np.array(features_list)
            self.mismatch = model.predict(features).flatten()
                   

        self.summary()

    def summary(self):
        if self.lam0 is not None:
            values_and_features = {
                'eta': self.eta,
                'q': self.q,
                'chieff': self.chi_eff,
                'chip': self.chi_p,
                'spin1': self.spin1,
                'spin2': self.spin2,
                'theta1': self.theta1,
                'theta2': self.theta2,
                'phi1': self.phi1,
                'phi2': self.phi2,
                'inplane1': self.inplane1,
                'inplane2': self.inplane2,
                'z1': self.z1,
                'z2': self.z2,
                'x1': self.x1,
                'x2': self.x2,
                'y1': self.y1,
                'y2': self.y2,
                'mismatch': self.mismatch
            }
        else:
            values_and_features = {
                'eta': self.eta,
                'q': self.q,
                'chieff': self.chi_eff,
                'chip': self.chi_p,
                'spin1': self.spin1,
                'spin2': self.spin2,
                'theta1': self.theta1,
                'theta2': self.theta2 ,
                'phi1': self.phi1,
                'phi2': self.phi2,
                'inplane1': self.inplane1,
                'inplane2': self.inplane2,
                'z1': self.z1,
                'z2': self.z2,
                'x1': self.x1,
                'x2': self.x2,
                'y1': self.y1,
                'y2': self.y2
            }    
        return values_and_features
    
        

    def perturb_lam(self):
        sampled_lams = []
       
        # Determine bounds from the data
        min_bounds = np.min(self.data, axis=0)
        max_bounds = np.max(self.data, axis=0)

        # Perturb the starting parameters within the bounds
        perturbed_params = [np.random.uniform(low=min_bound, high=max_bound) for min_bound, max_bound, start_val in zip(min_bounds, max_bounds, self.start)]

        # Convert the perturbed parameters to lambda representation
        lambda_sample = convert_to_lambda(*perturbed_params)

        return lambda_sample
    

class MapDegeneracyND:
  
    PARAM_STRINGS = {
        "eta": "$\eta$",
        "q": "$q$",
        "chieff": "$\chi_{\mathrm{eff}}$",
        "spin1": "$a_{1}$",
        "spin2": "$a_{2}$",
        "theta1": "$\\theta_1$",
        "theta2": "$\\theta_2$",
        "phi1": "$\\phi_1$",
        "phi2": "$\\phi_2$",
        "inplane1":"$\chi_{1\perp}",
        "inplane2":"$\chi_{2\perp}",
        "mismatch": "$\mathcal{MM}$",
        "chip": "$\chi_{p}$",
        "z1": '$a_{1z}$',
        "z2": '$a_{2z}$'
    }

    PARAM_BOUNDS = {
        "eta": (6/49, 0.25),
        "q": (1, 6),
        "chieff": (-1, 1),
        "spin1": (0, 1),
        "spin2": (0, 1),
        "theta1": (0, np.pi),
        "theta2": (0, np.pi),
        "phi1": (0, 2*np.pi),
        "phi2": (0, 2*np.pi),
        "mismatch": (0, 1),
        "chip": (0, 1),
        "inplane1":(0, 1),
        "inplane2":(0, 1),
        "z1": (-1,1),
        "z2": (-1,1),
        "x1": (-1,1),
        "y1": (-1,1),
        "x2": (-1,1),
        "y2": (-1,1)
    }
    

    def __init__(
        self,
        lam0,
        start,
        dimensions=["eta", "chieff"],
        stepsize = 1,
        max_iterations=1000,
        sample = 100000,
        fit_type="GMM",
        SNR = 4,
        precession = True,
        noalign = False,
        inplane = False
    ):
        #initial BBH parameters
        self.start = start
        self.origin = start.copy()
        self.lam0 = lam0
        self.lam = lam0.copy()
        self.precession = precession
        self.noalign =noalign
        self.inplane = inplane

        #mapping parameters
        self.dims = len(dimensions)
        self.dimensions = dimensions
        self.max_iterations = max_iterations
        self.fit_type = fit_type
        self.step_size = stepsize
        self.sample = sample

        self.SNR = SNR


        # Check if the provided dimensions list matches the dim argument
        if dimensions and len(dimensions) != self.dims:
            raise ValueError(f"The dimensions list provided does not match the specified dim of {dims}.")

        #values stored
        self.eigenvalues = []
        self.eigenvectors = []
        self.mismatch_from_reference = []
        self.mismatch_from_previous = []
        self.points = [np.array(self.start)]
        self.lams = [np.array(self.lam)]
        self.steps = []
        self.means = []
        

        self.ps, self.data_arrays = self.prepare_data()
        self.data_arrays = {k: v for k, v in self.data_arrays.items() if v is not None and len(v) > 0}



    def prepare_data(self):
        if not self.precession:
            parameterspace = ParameterSpace(sample = self.sample, precession = False)
        elif self.noalign:
            parameterspace = ParameterSpace(sample = self.sample, noalign = True)
        elif self.inplane:
            parameterspace = ParameterSpace(sample = self.sample, inplane = True)
        else:
            parameterspace = ParameterSpace(sample = self.sample)
        data = parameterspace.summary()
        data_arrays = {dim: self._fetch_array(data, dim) for dim in self.dimensions}
        
           # Print the data array tags and their dimensions
        print("Data Arrays Tags and Dimensions:")
        for key, value in data_arrays.items():
            print(f"Tag: {key}, Dimension: {len(value)}")
        return parameterspace, data_arrays

    def find_mismatch(self):
        features_list = []
        for lam in self.ps.lams:
            f = np.concatenate((self.lam, lam))
            features_list.append(f)
        features = np.array(features_list)
        mismatch = model.predict(features).flatten()
        return mismatch
    
    
    def _fetch_array(self, data, key):
        """Fetch data array using the provided key."""
        return data.get(key, [])

    def rejection_sampling(self, mismatch):    
        data_arrays = self.data_arrays
        # Calculate weights for rejection sampling based on mismatch values
        weights = np.exp(-self.SNR*self.SNR*mismatch*mismatch/2)
        data_arrays['mismatch'] = mismatch
        masked_data = {k: v for k, v in data_arrays.items()}
        samples = np.column_stack(tuple(masked_data.values()))
        
        #sampled_data_with_mismatch = rejection_sample(samples, weights)
        keep = weights > np.random.uniform(0, max(weights), weights.shape)
        sampled_data_with_mismatch = samples[keep]

        print("Size after rejection sampling:", len(sampled_data_with_mismatch))

        # Optionally, remove the mismatch column from the final data if it's no longer needed
        final_data = sampled_data_with_mismatch[:, :-1]
        
        return final_data, sampled_data_with_mismatch
    


    def fit_data(self, data_to_fit):
        if self.fit_type == "GMM":
            mean_prior = self.start
            print('mean prior', mean_prior)
            g = BayesianGaussianMixture(
                n_components=1, mean_prior =mean_prior, covariance_type="full")
            GMM = g.fit(data_to_fit)
            # print("Means:", GMM.means_)
            self.means.append(GMM.means_)
            return GMM
        elif self.fit_type == "PCA":
            # Adjust the PCA components to match the number of dimensions
            pca = PCA(n_components=self.dims)
            PCA_model = pca.fit(data_to_fit)
            # print("Means:", PCA_model.mean_)
            self.means.append(PCA_model.mean_)
            return PCA_model

                
                
    def find_direction(self, fitted_data, **kwargs):
        if self.fit_type == "GMM":
            cov = fitted_data.covariances_[0]
            eig_val, eig_vec = np.linalg.eigh(cov)
            order = np.argsort(eig_val)[::-1]
            eig_val = eig_val[order]
            eig_vec = eig_vec[:, order]
            
            for i, (val, vec) in enumerate(zip(eig_val, eig_vec)):
                print(f"Eigenvalue {i + 1}: {val}")
                print(f"Corresponding eigenvector: {vec}\n")
            
            largest = eig_vec.T[0]
            # print('largest eigenvector',largest)
            self.eigenvectors.append(largest)
     
        elif self.fit_type == "PCA":
            eig_val = fitted_data.explained_variance_
            eig_vec = fitted_data.components_
            self.eigenvectors.append(eig_vec[0]) 
       
        self.eigenvalues.append(eig_val)
        
        # Only keep largest
        eig_val = eig_val[0]
        if self.fit_type == "GMM":
            eig_vec = largest
        else:
            eig_vec = eig_vec[0]
        
        new_vector = np.sqrt(eig_val) * largest
        print('Computing new vector')                 
        if self.steps:
            previous_step = self.steps[-1]  # Get the last step

            # Check the direction of each component
            for i in range(len(new_vector)):
                if previous_step[i] * new_vector[i] < 0:  # If the product is negative, they are in opposite directions
                    new_vector[i] = -new_vector[i]  # Invert the direction of the new step's component

        self.steps.append(new_vector)
        
        return new_vector



    def run_mapping(self, backward = False):
        multplier = 1
        
        if backward:
            multplier = -1
            
        first_iteration = True 
        
        ps = self.ps
        
        for iteration in range(self.max_iterations):
            try:
                mismatch = self.find_mismatch()

                # 2. Fit the data (GMM or PCA)  
                
                data_to_fit, full_data = self.rejection_sampling(mismatch)
       

                fitted_data = self.fit_data(data_to_fit)
                
                # either FORWARD or BACKWARD
                direction = self.find_direction(fitted_data)*multplier
                print("mapping to direction:", direction)
                
                
                next_point = self.start + (direction / np.sqrt(self.dims))*self.step_size
                print("currently at ", self.start)
                print("proposed start at ", next_point)
                if not self.is_within_bounds(next_point):
                    print("Reached boundary. Stopping.")
                    break

                # set starting point for next iteration
                self.start = next_point

                # Update parameters, self.lam
                self.mismatch_at_point(direction)
            
                    
            except NoCorrelationException as e:
                print(e)
                print("No correlation found. Stopping mapping.")
                break
    
    def run_mapping_bothways(self):
        print('####################################')
        print('MAPPING FORWARD DIRECTION')
        self.run_mapping()
        print('####################################')        
        print('MAPPING BACKWARD DIRECTION')
        self.start = self.origin
        self.run_mapping(backward = True)
        print("Mapping complete.")
        
     

    def guess_lam(self, direction):
        if self.dims == 2:
            # if self.dimensions == ['q','inplane1']:
            #     q = self.start[0]
            #     eta = q/(1+q)**2
            #     x1 = self.start[1]
            #     lam = [eta,x1,0,0,0,0,0]
            if self.dimensions == ['eta','inplane1']:
                eta = self.start[0]
                x1 = self.start[1]
                lam = [eta,x1,0,0,0,0,0]
                
       
            # elif self.dimensions == ['eta','y1']:
            #     q = self.start[0]
            #     eta = q/(1+q)**2
            #     y1 = self.start[1]
            #     lam = [eta,0,y1,0,0,0,0]
                
            else:
                eta = self.start[0]
                z1 = self.start[1]
                z2 = self.start[1]
                lam = [eta,0,0,z1,0,0,z2]
          
        elif self.dims == 3: # default setting is ['q','z1','z2']
                q = self.start[0]
                eta = q/(1+q)**2
                z1 = self.start[1]
                z2 = self.start[2]
                lam = [eta,0,0,z1,0,0,z2]
        

        elif self.dims == 5:
            if self.dimensions == ['q','x1','x2','y1','y2']:
                q = self.start[0]
                x1, x2 = self.start[1], self.start[3]
                y1, y2 = self.start[2], self.start[4]
        
                eta = q / (1 + q)**2
                
                z1,z2 = self.lam0[3],self.lam0[6]
                
                lam = [eta, x1, y1, z1, x2, y2, z2]
                
            # elif self.dimensions == ['q', 'spin1', 'theta1','phi1','z2']:
            #     q = self.start[0]
            #     spin1 =  self.start[1]
            #     theta1 =  self.start[2]
            #     phi1 =  self.start[3]
            #     z2 = self.start[4]
            #     eta = q/(1+q)**2
            #     z1 = spin1*np.cos(theta1)
            #     inplane1 = spin1*np.sin(theta1)
            #     x1, y1 = inplane1*np.cos(phi1), inplane1*np.sin(phi1)
            #     lam = [eta, x1, y1, z1, 0, 0, z2]
                
            else:
                q = self.start[0]
                spin1, spin2 = self.start[1], self.start[2]
                theta1, theta2 = self.start[3], self.start[4]
                lam = convert_to_lambda(q,spin1,spin2,theta1,theta2)
                
        return lam
        

    def mismatch_at_point(self, direction):
        lam_old = self.lam
        self.lam = self.guess_lam(direction)
        
        print("new start at ", self.start)
        self.points.append(self.start)
        self.lams.append(self.lam)

        features1 = self.lam0 + self.lam
        features2 = lam_old + self.lam
        mismatch_prediction_ref = model.predict(features1).flatten()[0]
        mismatch_prediction_pre = model.predict(features2).flatten()[0]
        print(f"predicted mismatch from reference: {mismatch_prediction_ref}")
        print(f"predicted mismatch from previous: {mismatch_prediction_pre}")
        print("-----")

        self.mismatch_from_reference.append(mismatch_prediction_ref)
        self.mismatch_from_previous.append(mismatch_prediction_pre)

    def is_within_bounds(self, point):
        
            
        for dim_identifier, value in zip(self.dimensions, point):
            if not (
                self.PARAM_BOUNDS[dim_identifier][0]
                <= value
                <= self.PARAM_BOUNDS[dim_identifier][1]
            ):
                print(f"{dim_identifier} is out of bounds with value {value}")
                return False
        return True


class NoCorrelationException(Exception):
    pass

