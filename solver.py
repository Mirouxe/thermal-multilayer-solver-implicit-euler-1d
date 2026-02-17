"""
Solveur thermique 1D transitoire multicouche - Module réutilisable
==================================================================

Ce module contient le solveur de conduction thermique 1D implicite,
séparé de la définition des cas tests pour faciliter la validation.

Équation résolue:
    ρ(T) * cp(T) * ∂T/∂t = ∂/∂x [k(T) * ∂T/∂x]

Schéma numérique:
    - Temporel: Euler implicite (inconditionnellement stable)
    - Spatial: Différences finies centrées

Conditions limites supportées:
    - Gauche/Droite: 
        * Convection (Robin): -k·∂T/∂x = h·(T - T_inf)
        * Flux imposé (Neumann): -k·∂T/∂x = q
        * Isotherme (Dirichlet): T = T_imposé
        * Adiabatique: ∂T/∂x = 0
        * Rayonnement: -k·∂T/∂x = σ·ε·(T⁴ - T_s⁴)
        * Convection + Rayonnement combinés
        
Constantes physiques:
    σ (Stefan-Boltzmann) = 5.670374419e-8 W/(m²·K⁴)

OPTIMISATIONS APPLIQUÉES:
    1. Solveur tridiagonal (Thomas algorithm) au lieu de np.linalg.solve - O(n) vs O(n³)
    2. Vectorisation de get_properties_at_nodes - évite boucle Python
    3. Vectorisation de la construction des nœuds intérieurs
    4. Pré-calcul des indices de matériaux par couche
"""

# Constante de Stefan-Boltzmann [W/(m²·K⁴)]
STEFAN_BOLTZMANN = 5.670374419e-8

import numpy as np
from scipy.linalg import solve_banded  # [OPTIM 1] Solveur tridiagonal O(n)


class ThermalSolver1D:
    """
    Solveur de conduction thermique 1D transitoire multicouche.
    
    Attributs:
        layers: Liste de dictionnaires définissant les couches
        material_data: Dictionnaire des propriétés des matériaux
        Nx: Nombre de nœuds spatiaux
        dx: Pas spatial [m]
        x: Coordonnées des nœuds [m]
        L_total: Longueur totale du domaine [m]
    """
    
    def __init__(self, layers, material_data, Nx=101):
        """
        Initialise le solveur.
        
        Args:
            layers: Liste de dict {'material': str, 'thickness': float}
            material_data: Dict des propriétés thermiques par matériau
            Nx: Nombre de nœuds spatiaux
        """
        self.layers = layers
        self.material_data = material_data
        self.Nx = Nx
        
        # Calcul du maillage
        self.L_total = sum(layer['thickness'] for layer in layers)
        self.dx = self.L_total / (Nx - 1)
        self.x = np.linspace(0, self.L_total, Nx)
        
        # Déterminer l'indice de couche pour chaque nœud
        cum_thick = np.cumsum([layer['thickness'] for layer in layers])
        self.layer_index = np.searchsorted(cum_thick, self.x)
        self.layer_index = np.clip(self.layer_index, 0, len(layers) - 1)
        
        # [OPTIM 4] Pré-calcul: liste des matériaux par nœud (évite lookup répété)
        self.node_materials = [layers[idx]['material'] for idx in self.layer_index]
        
        # [OPTIM 4] Pré-calcul: grouper les nœuds par matériau pour vectorisation
        self._precompute_material_groups()
    
    def _precompute_material_groups(self):
        """
        [OPTIM 4] Pré-calcule les groupes de nœuds par matériau.
        Permet la vectorisation de l'interpolation des propriétés.
        """
        self.material_groups = {}  # {material_name: array of node indices}
        for mat_name in set(self.node_materials):
            indices = np.array([i for i, m in enumerate(self.node_materials) if m == mat_name])
            self.material_groups[mat_name] = indices
    
    def interp_property(self, mat_props, T):
        """
        Interpole les propriétés thermiques (k, rho, cp) à la température T.
        
        Args:
            mat_props: Dict contenant 'T', 'k', 'rho', 'cp'
            T: Température d'interpolation [K] (scalaire)
            
        Returns:
            Tuple (k, rho, cp) interpolés (scalaires)
        """
        T_tab = mat_props['T']
        k_tab = mat_props['k']
        rho_tab = mat_props['rho']
        cp_tab = mat_props['cp']
        T_val = float(T)
        k = float(np.interp(T_val, T_tab, k_tab))
        rho = float(np.interp(T_val, T_tab, rho_tab))
        cp = float(np.interp(T_val, T_tab, cp_tab))
        return k, rho, cp
    
    def get_properties_at_nodes(self, T):
        """
        [OPTIM 2] Calcule les propriétés thermiques à chaque nœud - VERSION VECTORISÉE.
        
        Au lieu de boucler sur chaque nœud individuellement, on groupe les nœuds
        par matériau et on applique np.interp sur des arrays complets.
        
        Args:
            T: Champ de température actuel [K]
            
        Returns:
            Tuple (k_nodes, rho_nodes, cp_nodes)
        """
        k_nodes = np.zeros(self.Nx)
        rho_nodes = np.zeros(self.Nx)
        cp_nodes = np.zeros(self.Nx)
        
        # [OPTIM 2] Vectorisation: traiter tous les nœuds d'un même matériau en une fois
        for mat_name, indices in self.material_groups.items():
            mat_props = self.material_data[mat_name]
            T_tab = mat_props['T']
            k_tab = mat_props['k']
            rho_tab = mat_props['rho']
            cp_tab = mat_props['cp']
            
            # Interpolation vectorisée sur tous les nœuds de ce matériau
            T_at_nodes = T[indices]  # [OPTIM 2] Extraction vectorisée des températures
            k_nodes[indices] = np.interp(T_at_nodes, T_tab, k_tab)    # [OPTIM 2] Interp vectorisée
            rho_nodes[indices] = np.interp(T_at_nodes, T_tab, rho_tab)  # [OPTIM 2] Interp vectorisée
            cp_nodes[indices] = np.interp(T_at_nodes, T_tab, cp_tab)   # [OPTIM 2] Interp vectorisée
        
        return k_nodes, rho_nodes, cp_nodes
    
    def build_system_tridiagonal(self, T, dt, bc_left, bc_right):
        """
        [OPTIM 1 & 3] Construit le système tridiagonal pour le schéma implicite.
        
        Format banded pour scipy.linalg.solve_banded((1,1), ab, b):
        Pour une matrice tridiagonale A où A[i,i-1]=lower, A[i,i]=diag, A[i,i+1]=upper:
        
            ab[0, j] = A[j-1, j]  (upper diagonal, first element unused)
            ab[1, j] = A[j, j]    (main diagonal)  
            ab[2, j] = A[j+1, j]  (lower diagonal, last element unused)
        
        Args:
            T: Champ de température actuel [K]
            dt: Pas de temps [s]
            bc_left, bc_right: Conditions limites
            
        Returns:
            Tuple (ab, b) où ab est la matrice banded (3, Nx) et b le vecteur RHS
        """
        Nx = self.Nx
        dx = self.dx
        
        # [OPTIM 1] Stockage des 3 diagonales
        diag_lower = np.zeros(Nx)   # A[i, i-1] stocké à position i
        diag_main = np.zeros(Nx)    # A[i, i]
        diag_upper = np.zeros(Nx)   # A[i, i+1] stocké à position i
        b = np.zeros(Nx)
        
        # Propriétés aux nœuds (vectorisé via [OPTIM 2])
        k_nodes, rho_nodes, cp_nodes = self.get_properties_at_nodes(T)
        
        # === Condition limite gauche (i=0) ===
        k0p = 0.5 * (k_nodes[0] + k_nodes[1])
        
        if bc_left['type'] == 'dirichlet':
            diag_main[0] = 1.0
            diag_upper[0] = 0.0
            b[0] = bc_left['T']
        elif bc_left['type'] == 'convection':
            h = bc_left['h']
            T_inf = bc_left['T_inf']
            coef = dt / (rho_nodes[0] * cp_nodes[0] * dx / 2)
            diag_main[0] = 1 + coef * (k0p / dx + h)
            diag_upper[0] = -coef * (k0p / dx)
            b[0] = T[0] + coef * h * T_inf
        elif bc_left['type'] == 'flux':
            q = bc_left['q']
            coef = dt / (rho_nodes[0] * cp_nodes[0] * dx / 2)
            diag_main[0] = 1 + coef * (k0p / dx)
            diag_upper[0] = -coef * (k0p / dx)
            b[0] = T[0] + coef * q
        elif bc_left['type'] == 'radiation':
            epsilon = bc_left['epsilon']
            T_s = bc_left['T_s']
            T0 = T[0]
            h_rad = 4 * STEFAN_BOLTZMANN * epsilon * T0**3
            q_rad_old = STEFAN_BOLTZMANN * epsilon * (T_s**4 - T0**4)
            T_rad_eq = T0 + q_rad_old / (h_rad + 1e-10)
            coef = dt / (rho_nodes[0] * cp_nodes[0] * dx / 2)
            diag_main[0] = 1 + coef * (k0p / dx + h_rad)
            diag_upper[0] = -coef * (k0p / dx)
            b[0] = T[0] + coef * h_rad * T_rad_eq
        elif bc_left['type'] == 'convection_radiation':
            h_conv = bc_left['h']
            T_inf = bc_left['T_inf']
            epsilon = bc_left['epsilon']
            T_s = bc_left['T_s']
            T0 = T[0]
            h_rad = 4 * STEFAN_BOLTZMANN * epsilon * T0**3
            q_rad_old = STEFAN_BOLTZMANN * epsilon * (T_s**4 - T0**4)
            T_rad_eq = T0 + q_rad_old / (h_rad + 1e-10)
            h_total = h_conv + h_rad
            T_eq = (h_conv * T_inf + h_rad * T_rad_eq) / (h_total + 1e-10)
            coef = dt / (rho_nodes[0] * cp_nodes[0] * dx / 2)
            diag_main[0] = 1 + coef * (k0p / dx + h_total)
            diag_upper[0] = -coef * (k0p / dx)
            b[0] = T[0] + coef * h_total * T_eq
        else:
            raise ValueError(f"Type de CL gauche inconnu: {bc_left['type']}")
        
        # === [OPTIM 3] Nœuds intérieurs (1 ≤ i ≤ Nx-2) - VERSION VECTORISÉE ===
        idx = np.arange(1, Nx - 1)  # [OPTIM 3] Indices vectorisés
        
        kim = 0.5 * (k_nodes[idx] + k_nodes[idx-1])  # [OPTIM 3] k_{i-1/2} vectorisé
        kip = 0.5 * (k_nodes[idx] + k_nodes[idx+1])  # [OPTIM 3] k_{i+1/2} vectorisé
        
        coef = dt / (rho_nodes[idx] * cp_nodes[idx] * dx**2)  # [OPTIM 3] Vectorisé
        
        diag_lower[idx] = -coef * kim              # [OPTIM 3] A[i, i-1]
        diag_main[idx] = 1 + coef * (kim + kip)    # [OPTIM 3] A[i, i]
        diag_upper[idx] = -coef * kip              # [OPTIM 3] A[i, i+1]
        b[idx] = T[idx]                            # [OPTIM 3] RHS vectorisé
        
        # === Condition limite droite (i=Nx-1) ===
        i = Nx - 1
        kim_last = 0.5 * (k_nodes[i] + k_nodes[i-1])
        
        if bc_right['type'] == 'dirichlet':
            diag_main[i] = 1.0
            diag_lower[i] = 0.0
            b[i] = bc_right['T']
        elif bc_right['type'] == 'adiabatic':
            coef = dt / (rho_nodes[i] * cp_nodes[i] * dx / 2)
            diag_lower[i] = -coef * (kim_last / dx)
            diag_main[i] = 1 + coef * (kim_last / dx)
            b[i] = T[i]
        elif bc_right['type'] == 'convection':
            h = bc_right['h']
            T_inf = bc_right['T_inf']
            coef = dt / (rho_nodes[i] * cp_nodes[i] * dx / 2)
            diag_lower[i] = -coef * (kim_last / dx)
            diag_main[i] = 1 + coef * (kim_last / dx + h)
            b[i] = T[i] + coef * h * T_inf
        elif bc_right['type'] == 'radiation':
            epsilon = bc_right['epsilon']
            T_s = bc_right['T_s']
            Ti = T[i]
            h_rad = 4 * STEFAN_BOLTZMANN * epsilon * Ti**3
            q_rad_old = STEFAN_BOLTZMANN * epsilon * (T_s**4 - Ti**4)
            T_rad_eq = Ti + q_rad_old / (h_rad + 1e-10)
            coef = dt / (rho_nodes[i] * cp_nodes[i] * dx / 2)
            diag_lower[i] = -coef * (kim_last / dx)
            diag_main[i] = 1 + coef * (kim_last / dx + h_rad)
            b[i] = T[i] + coef * h_rad * T_rad_eq
        elif bc_right['type'] == 'convection_radiation':
            h_conv = bc_right['h']
            T_inf = bc_right['T_inf']
            epsilon = bc_right['epsilon']
            T_s = bc_right['T_s']
            Ti = T[i]
            h_rad = 4 * STEFAN_BOLTZMANN * epsilon * Ti**3
            q_rad_old = STEFAN_BOLTZMANN * epsilon * (T_s**4 - Ti**4)
            T_rad_eq = Ti + q_rad_old / (h_rad + 1e-10)
            h_total = h_conv + h_rad
            T_eq = (h_conv * T_inf + h_rad * T_rad_eq) / (h_total + 1e-10)
            coef = dt / (rho_nodes[i] * cp_nodes[i] * dx / 2)
            diag_lower[i] = -coef * (kim_last / dx)
            diag_main[i] = 1 + coef * (kim_last / dx + h_total)
            b[i] = T[i] + coef * h_total * T_eq
        elif bc_right['type'] == 'flux':
            q = bc_right['q']
            coef = dt / (rho_nodes[i] * cp_nodes[i] * dx / 2)
            diag_lower[i] = -coef * (kim_last / dx)
            diag_main[i] = 1 + coef * (kim_last / dx)
            b[i] = T[i] - coef * q
        else:
            raise ValueError(f"Type de CL droite inconnu: {bc_right['type']}")
        
        # [OPTIM 1] Assemblage au format banded pour solve_banded((1,1), ab, b)
        # ab[0, j] = A[j-1, j] = diag_upper[j-1] (décalé vers la droite, premier élément inutilisé)
        # ab[1, j] = A[j, j] = diag_main[j]
        # ab[2, j] = A[j+1, j] = diag_lower[j+1] (décalé vers la gauche, dernier élément inutilisé)
        ab = np.zeros((3, Nx))
        ab[0, 1:] = diag_upper[:-1]   # [OPTIM 1] Upper: décalé d'un cran vers la droite
        ab[1, :] = diag_main          # [OPTIM 1] Main diagonal
        ab[2, :-1] = diag_lower[1:]   # [OPTIM 1] Lower: décalé d'un cran vers la gauche
        
        return ab, b
    
    def build_system(self, T, dt, bc_left, bc_right):
        """
        Construit le système linéaire A*T_new = b pour le schéma implicite.
        (Version maintenue pour compatibilité avec les tests)
        
        Returns:
            Tuple (A, b) du système linéaire (matrice pleine)
        """
        Nx = self.Nx
        dx = self.dx
        
        A = np.zeros((Nx, Nx))
        b = np.zeros(Nx)
        
        k_nodes, rho_nodes, cp_nodes = self.get_properties_at_nodes(T)
        
        # === Condition limite gauche (i=0) ===
        if bc_left['type'] == 'dirichlet':
            A[0, 0] = 1.0
            b[0] = bc_left['T']
        elif bc_left['type'] == 'convection':
            h = bc_left['h']
            T_inf = bc_left['T_inf']
            k0p = 0.5 * (k_nodes[0] + k_nodes[1])
            rho0 = rho_nodes[0]
            cp0 = cp_nodes[0]
            coef = dt / (rho0 * cp0 * dx / 2)
            A[0, 0] = 1 + coef * (k0p / dx + h)
            A[0, 1] = -coef * (k0p / dx)
            b[0] = T[0] + coef * h * T_inf
        elif bc_left['type'] == 'flux':
            q = bc_left['q']
            k0p = 0.5 * (k_nodes[0] + k_nodes[1])
            rho0 = rho_nodes[0]
            cp0 = cp_nodes[0]
            coef = dt / (rho0 * cp0 * dx / 2)
            A[0, 0] = 1 + coef * (k0p / dx)
            A[0, 1] = -coef * (k0p / dx)
            b[0] = T[0] + coef * q
        elif bc_left['type'] == 'radiation':
            epsilon = bc_left['epsilon']
            T_s = bc_left['T_s']
            T0 = T[0]
            h_rad = 4 * STEFAN_BOLTZMANN * epsilon * T0**3
            q_rad_old = STEFAN_BOLTZMANN * epsilon * (T_s**4 - T0**4)
            T_rad_eq = T0 + q_rad_old / (h_rad + 1e-10)
            k0p = 0.5 * (k_nodes[0] + k_nodes[1])
            rho0 = rho_nodes[0]
            cp0 = cp_nodes[0]
            coef = dt / (rho0 * cp0 * dx / 2)
            A[0, 0] = 1 + coef * (k0p / dx + h_rad)
            A[0, 1] = -coef * (k0p / dx)
            b[0] = T[0] + coef * h_rad * T_rad_eq
        elif bc_left['type'] == 'convection_radiation':
            h_conv = bc_left['h']
            T_inf = bc_left['T_inf']
            epsilon = bc_left['epsilon']
            T_s = bc_left['T_s']
            T0 = T[0]
            h_rad = 4 * STEFAN_BOLTZMANN * epsilon * T0**3
            q_rad_old = STEFAN_BOLTZMANN * epsilon * (T_s**4 - T0**4)
            T_rad_eq = T0 + q_rad_old / (h_rad + 1e-10)
            h_total = h_conv + h_rad
            T_eq = (h_conv * T_inf + h_rad * T_rad_eq) / (h_total + 1e-10)
            k0p = 0.5 * (k_nodes[0] + k_nodes[1])
            rho0 = rho_nodes[0]
            cp0 = cp_nodes[0]
            coef = dt / (rho0 * cp0 * dx / 2)
            A[0, 0] = 1 + coef * (k0p / dx + h_total)
            A[0, 1] = -coef * (k0p / dx)
            b[0] = T[0] + coef * h_total * T_eq
        else:
            raise ValueError(f"Type de CL gauche inconnu: {bc_left['type']}")
        
        # === Nœuds intérieurs ===
        for i in range(1, Nx - 1):
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])
            kip = 0.5 * (k_nodes[i] + k_nodes[i+1])
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            coef = dt / (rhoi * cpi * dx**2)
            A[i, i-1] = -coef * kim
            A[i, i] = 1 + coef * (kim + kip)
            A[i, i+1] = -coef * kip
            b[i] = T[i]
        
        # === Condition limite droite ===
        i = Nx - 1
        if bc_right['type'] == 'dirichlet':
            A[i, i] = 1.0
            b[i] = bc_right['T']
        elif bc_right['type'] == 'adiabatic':
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            coef = dt / (rhoi * cpi * dx / 2)
            A[i, i-1] = -coef * (kim / dx)
            A[i, i] = 1 + coef * (kim / dx)
            b[i] = T[i]
        elif bc_right['type'] == 'convection':
            h = bc_right['h']
            T_inf = bc_right['T_inf']
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            coef = dt / (rhoi * cpi * dx / 2)
            A[i, i-1] = -coef * (kim / dx)
            A[i, i] = 1 + coef * (kim / dx + h)
            b[i] = T[i] + coef * h * T_inf
        elif bc_right['type'] == 'radiation':
            epsilon = bc_right['epsilon']
            T_s = bc_right['T_s']
            Ti = T[i]
            h_rad = 4 * STEFAN_BOLTZMANN * epsilon * Ti**3
            q_rad_old = STEFAN_BOLTZMANN * epsilon * (T_s**4 - Ti**4)
            T_rad_eq = Ti + q_rad_old / (h_rad + 1e-10)
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            coef = dt / (rhoi * cpi * dx / 2)
            A[i, i-1] = -coef * (kim / dx)
            A[i, i] = 1 + coef * (kim / dx + h_rad)
            b[i] = T[i] + coef * h_rad * T_rad_eq
        elif bc_right['type'] == 'convection_radiation':
            h_conv = bc_right['h']
            T_inf = bc_right['T_inf']
            epsilon = bc_right['epsilon']
            T_s = bc_right['T_s']
            Ti = T[i]
            h_rad = 4 * STEFAN_BOLTZMANN * epsilon * Ti**3
            q_rad_old = STEFAN_BOLTZMANN * epsilon * (T_s**4 - Ti**4)
            T_rad_eq = Ti + q_rad_old / (h_rad + 1e-10)
            h_total = h_conv + h_rad
            T_eq = (h_conv * T_inf + h_rad * T_rad_eq) / (h_total + 1e-10)
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            coef = dt / (rhoi * cpi * dx / 2)
            A[i, i-1] = -coef * (kim / dx)
            A[i, i] = 1 + coef * (kim / dx + h_total)
            b[i] = T[i] + coef * h_total * T_eq
        elif bc_right['type'] == 'flux':
            q = bc_right['q']
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            coef = dt / (rhoi * cpi * dx / 2)
            A[i, i-1] = -coef * (kim / dx)
            A[i, i] = 1 + coef * (kim / dx)
            b[i] = T[i] - coef * q
        else:
            raise ValueError(f"Type de CL droite inconnu: {bc_right['type']}")
        
        return A, b
    
    def solve_step(self, T, dt, bc_left, bc_right):
        """
        [OPTIM 1] Résout un pas de temps avec solveur tridiagonal.
        
        Utilise solve_banded de scipy qui implémente l'algorithme de Thomas
        en O(n) au lieu de O(n³) pour np.linalg.solve sur matrice pleine.
        
        Args:
            T: Champ de température actuel [K]
            dt: Pas de temps [s]
            bc_left: Condition limite gauche
            bc_right: Condition limite droite
            
        Returns:
            Nouveau champ de température [K]
        """
        # [OPTIM 1] Construction du système en format banded (3 diagonales)
        ab, b = self.build_system_tridiagonal(T, dt, bc_left, bc_right)
        
        # [OPTIM 1] Résolution tridiagonale O(n) au lieu de O(n³)
        # Format banded: ab[0] = diag supérieure, ab[1] = diag principale, ab[2] = diag inférieure
        # (l, u) = (1, 1) signifie 1 diagonale sous et 1 au-dessus de la principale
        T_new = solve_banded((1, 1), ab, b)
        
        return T_new
    
    def solve(self, T_init, t_end, dt, bc_left_func, bc_right_func, 
              save_every=1, verbose=False):
        """
        Résout le problème transitoire complet.
        
        Args:
            T_init: Condition initiale (scalaire ou array)
            t_end: Temps final [s]
            dt: Pas de temps [s]
            bc_left_func: Fonction(t) retournant le dict de CL gauche
            bc_right_func: Fonction(t) retournant le dict de CL droite
            save_every: Sauvegarder tous les N pas de temps
            verbose: Afficher la progression
            
        Returns:
            Dict contenant:
                'time': Array des temps sauvegardés
                'T': Array 2D (nt_saved, Nx) des températures
                'x': Coordonnées spatiales
        """
        # Initialisation
        if np.isscalar(T_init):
            T = np.ones(self.Nx) * T_init
        else:
            T = np.array(T_init).copy()
        
        nt = int(t_end / dt) + 1
        time_steps = np.linspace(0, t_end, nt)
        
        # Stockage - utiliser set pour lookup O(1) au lieu de list O(n)
        save_indices = set(range(0, nt, save_every))  # [OPTIM] set au lieu de list
        if (nt - 1) not in save_indices:
            save_indices.add(nt - 1)
        
        T_hist = []
        time_hist = []
        
        for n, t in enumerate(time_steps):
            if n in save_indices:  # [OPTIM] Lookup O(1) avec set
                T_hist.append(T.copy())
                time_hist.append(t)
            
            if n < nt - 1:
                bc_left = bc_left_func(t + dt)
                bc_right = bc_right_func(t + dt)
                T = self.solve_step(T, dt, bc_left, bc_right)
            
            if verbose and n % 100 == 0:
                print(f"  t = {t:.2f} s, T_min = {T.min():.2f} K, T_max = {T.max():.2f} K")
        
        return {
            'time': np.array(time_hist),
            'T': np.array(T_hist),
            'x': self.x.copy()
        }


def create_constant_material(k, rho, cp, name='constant'):
    """
    Crée un matériau avec des propriétés constantes.
    
    Args:
        k: Conductivité thermique [W/(m·K)]
        rho: Masse volumique [kg/m³]
        cp: Capacité thermique massique [J/(kg·K)]
        name: Nom du matériau
        
    Returns:
        Dict compatible avec le solveur
    """
    return {
        name: {
            'T': np.array([0, 1000]),
            'k': np.array([k, k]),
            'rho': np.array([rho, rho]),
            'cp': np.array([cp, cp])
        }
    }


def create_single_layer(L, material_name):
    """
    Crée une configuration monocouche.
    
    Args:
        L: Épaisseur [m]
        material_name: Nom du matériau
        
    Returns:
        Liste de couches (1 élément)
    """
    return [{'material': material_name, 'thickness': L}]


# ============================================================================
# Fonctions utilitaires pour l'analyse des résultats
# ============================================================================

def compute_error_norms(T_num, T_ref):
    """
    Calcule les normes d'erreur L2 et Linf.
    """
    err = T_num - T_ref
    L2 = np.sqrt(np.mean(err**2))
    Linf = np.max(np.abs(err))
    
    T_range = np.max(T_ref) - np.min(T_ref)
    if T_range > 1e-10:
        L2_rel = L2 / T_range
        Linf_rel = Linf / T_range
    else:
        L2_rel = L2 / (np.mean(np.abs(T_ref)) + 1e-10)
        Linf_rel = Linf / (np.mean(np.abs(T_ref)) + 1e-10)
    
    return {
        'L2': L2,
        'Linf': Linf,
        'L2_rel': L2_rel,
        'Linf_rel': Linf_rel
    }


def compute_heat_flux(T, x, k):
    """
    Calcule le flux de chaleur q = -k * dT/dx.
    """
    dTdx = np.diff(T) / np.diff(x)
    if np.isscalar(k):
        k_interf = k
    else:
        k_interf = 0.5 * (k[:-1] + k[1:])
    return -k_interf * dTdx


def compute_radiation_flux(T_surface, T_surroundings, epsilon):
    """
    Calcule le flux radiatif selon la loi de Stefan-Boltzmann.
    """
    return STEFAN_BOLTZMANN * epsilon * (T_surface**4 - T_surroundings**4)


def compute_radiation_coefficient(T_surface, epsilon):
    """
    Calcule le coefficient de transfert radiatif linéarisé.
    """
    return 4 * STEFAN_BOLTZMANN * epsilon * T_surface**3


def compute_energy_balance(T, T_prev, rho, cp, dx, dt, q_left, q_right):
    """
    Vérifie le bilan d'énergie sur un pas de temps.
    """
    dE = np.sum(rho * cp * (T - T_prev) * dx)
    Q_in = (q_left + q_right) * dt
    
    return {
        'dE_stored': dE,
        'Q_in': Q_in,
        'residual': dE - Q_in,
        'residual_rel': (dE - Q_in) / (np.abs(Q_in) + 1e-10)
    }
