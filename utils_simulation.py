"""
Fonctions utilitaires pour la simulation
========================================

Ce fichier contient toutes les fonctions de:
- Chargement des données (tables h, essais)
- Interpolation
- Calcul des conditions limites
- Exécution des simulations
- Gestion des dossiers de sortie

OPTIMISATIONS APPLIQUÉES:
- [OPTIM 5] Parallélisation des simulations avec multiprocessing
"""

import numpy as np
import os
import shutil
from scipy.interpolate import LinearNDInterpolator
from datetime import datetime
from multiprocessing import Pool, cpu_count  # [OPTIM 5] Import pour parallélisation

from solver import ThermalSolver1D
from material_library import get_materials
from param_simu import (
    TABLES_DIR, ESSAIS_DIR, RESULTS_DIR, N_SAVE_POINTS,
    TABLE_COLUMNS, ESSAI_COLUMNS,
    ZONE_TO_EMPILEMENT, TABLE_FILE_PREFIX
)
from empilement_library import get_empilement


# [OPTIM 5] Nombre de processus parallèles (par défaut: nombre de CPU - 1)
N_WORKERS = max(1, cpu_count() - 1)


# ============================================================================
# 1. GESTION DES DOSSIERS DE SORTIE
# ============================================================================

def create_output_directory(base_dir=None):
    """
    Crée le dossier de sortie pour cette simulation.
    
    Structure créée:
        results/simu_YYYYMMDD_HHMMSS/
            ├── result/      # Résultats CSV
            ├── visu/        # Visualisations PNG
            └── calcul/      # Copie des scripts
    
    Args:
        base_dir: Dossier de base (défaut: RESULTS_DIR)
    
    Returns:
        Dict avec les chemins:
            - 'root': Dossier racine de la simulation
            - 'result': Dossier des résultats
            - 'visu': Dossier des visualisations
            - 'calcul': Dossier des scripts
    """
    if base_dir is None:
        base_dir = RESULTS_DIR
    
    # Créer le dossier de base si nécessaire
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Nom du dossier avec date/heure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    simu_name = f"simu_{timestamp}"
    root_dir = os.path.join(base_dir, simu_name)
    
    # Créer les sous-dossiers
    result_dir = os.path.join(root_dir, 'result')
    visu_dir = os.path.join(root_dir, 'visu')
    calcul_dir = os.path.join(root_dir, 'calcul')
    
    os.makedirs(result_dir)
    os.makedirs(visu_dir)
    os.makedirs(calcul_dir)
    
    return {
        'root': root_dir,
        'result': result_dir,
        'visu': visu_dir,
        'calcul': calcul_dir,
        'timestamp': timestamp
    }


def copy_scripts_to_calcul(calcul_dir, workspace_dir='.'):
    """
    Copie tous les scripts nécessaires dans le dossier calcul.
    
    Fichiers copiés:
        - Scripts Python principaux
        - Dossier tables/ (tables h(M,A))
        - Dossier essais/ (fichiers d'essai)
    
    Args:
        calcul_dir: Dossier de destination
        workspace_dir: Dossier de travail source
    """
    # Liste des scripts Python à copier
    scripts = [
        'multi_simulation.py',
        'param_simu.py',
        'empilement_library.py',
        'utils_simulation.py',
        'post.py',
        'solver.py',
        'material_library.py',
    ]
    
    # Copier les scripts
    for script in scripts:
        src = os.path.join(workspace_dir, script)
        if os.path.exists(src):
            dst = os.path.join(calcul_dir, script)
            shutil.copy2(src, dst)
    
    # Copier le dossier tables/
    src_tables = os.path.join(workspace_dir, TABLES_DIR)
    if os.path.exists(src_tables):
        dst_tables = os.path.join(calcul_dir, TABLES_DIR)
        shutil.copytree(src_tables, dst_tables)
    
    # Copier le dossier essais/
    src_essais = os.path.join(workspace_dir, ESSAIS_DIR)
    if os.path.exists(src_essais):
        dst_essais = os.path.join(calcul_dir, ESSAIS_DIR)
        shutil.copytree(src_essais, dst_essais)
    
    # Copier requirements.txt si présent
    req_file = os.path.join(workspace_dir, 'requirements.txt')
    if os.path.exists(req_file):
        shutil.copy2(req_file, os.path.join(calcul_dir, 'requirements.txt'))


# ============================================================================
# 2. CHARGEMENT DES TABLES h(M, A)
# ============================================================================

def load_h_table(filepath):
    """
    Charge une table (M, A, h) depuis un fichier CSV.
    
    Les noms des colonnes sont définis dans param_simu.TABLE_COLUMNS:
        - TABLE_COLUMNS['M']: nom de la colonne M (ex: 'Mach')
        - TABLE_COLUMNS['A']: nom de la colonne A (ex: 'Alph')
        - TABLE_COLUMNS['h']: nom de la colonne h (ex: 'Ste')
    
    Args:
        filepath: Chemin du fichier CSV
    
    Returns:
        Dict avec:
            - 'M', 'A', 'h': Arrays des valeurs
            - 'interpolator': Fonction d'interpolation 2D
            - 'M_range', 'A_range': Bornes du domaine
    """
    # Lire l'en-tête pour trouver les indices des colonnes
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
    
    # Trouver les indices des colonnes
    col_M = TABLE_COLUMNS['M']
    col_A = TABLE_COLUMNS['A']
    col_h = TABLE_COLUMNS['h']
    
    try:
        idx_M = header.index(col_M)
        idx_A = header.index(col_A)
        idx_h = header.index(col_h)
    except ValueError as e:
        raise ValueError(
            f"Colonne non trouvée dans {filepath}. "
            f"Attendu: {col_M}, {col_A}, {col_h}. "
            f"Trouvé: {header}"
        )
    
    # Charger les données
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    M = data[:, idx_M]
    A = data[:, idx_A]
    h = data[:, idx_h]
    
    # Créer l'interpolateur
    points = np.column_stack([M, A])
    interpolator = LinearNDInterpolator(points, h)
    
    return {
        'M': M,
        'A': A,
        'h': h,
        'interpolator': interpolator,
        'M_range': (M.min(), M.max()),
        'A_range': (A.min(), A.max()),
        'filepath': filepath
    }


def get_zone_from_filename(filename):
    """
    Extrait le nom de la zone depuis le nom du fichier.
    
    Format attendu: data_<nom_zone>.csv
    Exemple: data_Zone1.csv -> 'Zone1'
    """
    basename = os.path.basename(filename)
    
    if not basename.startswith(TABLE_FILE_PREFIX):
        raise ValueError(
            f"Le fichier '{basename}' ne commence pas par '{TABLE_FILE_PREFIX}'"
        )
    
    zone_name = basename[len(TABLE_FILE_PREFIX):]
    zone_name = zone_name.rsplit('.', 1)[0]
    
    return zone_name


def get_empilement_for_zone(zone_name):
    """
    Récupère l'empilement associé à une zone.
    """
    if zone_name not in ZONE_TO_EMPILEMENT:
        raise ValueError(
            f"Zone '{zone_name}' non trouvée dans ZONE_TO_EMPILEMENT. "
            f"Zones disponibles: {list(ZONE_TO_EMPILEMENT.keys())}"
        )
    
    emp_name = ZONE_TO_EMPILEMENT[zone_name]
    emp = get_empilement(emp_name)
    
    if emp is None:
        raise ValueError(
            f"Empilement '{emp_name}' non trouvé pour la zone '{zone_name}'"
        )
    
    return emp


def load_all_h_tables(tables_dir=None):
    """
    Charge toutes les tables h(M, A) depuis le dossier.
    
    Returns:
        Tuple (h_tables, zone_empilements)
    """
    if tables_dir is None:
        tables_dir = TABLES_DIR
    
    h_tables = {}
    zone_empilements = {}
    
    if not os.path.exists(tables_dir):
        raise FileNotFoundError(f"Dossier non trouvé: {tables_dir}")
    
    table_files = [f for f in os.listdir(tables_dir) 
                   if f.startswith(TABLE_FILE_PREFIX) and f.endswith('.csv')]
    
    if not table_files:
        raise FileNotFoundError(
            f"Aucun fichier {TABLE_FILE_PREFIX}*.csv trouvé dans {tables_dir}"
        )
    
    for filename in sorted(table_files):
        filepath = os.path.join(tables_dir, filename)
        zone_name = get_zone_from_filename(filename)
        emp = get_empilement_for_zone(zone_name)
        
        h_tables[zone_name] = load_h_table(filepath)
        zone_empilements[zone_name] = emp
        
        print(f"   • {zone_name} ({filename}) -> {emp['name']}")
    
    return h_tables, zone_empilements


def get_h_from_table(h_table, M, A):
    """
    Interpole h à partir de la table pour les valeurs M et A.
    """
    h = h_table['interpolator'](M, A)
    
    if np.isnan(h):
        M_clamped = np.clip(M, *h_table['M_range'])
        A_clamped = np.clip(A, *h_table['A_range'])
        h = h_table['interpolator'](M_clamped, A_clamped)
        
        if np.isnan(h):
            distances = (h_table['M'] - M)**2 + (h_table['A'] - A)**2
            idx = np.argmin(distances)
            h = h_table['h'][idx]
    
    return float(h)


# ============================================================================
# 3. CHARGEMENT DES FICHIERS D'ESSAI
# ============================================================================

def load_test_csv(filepath):
    """
    Charge un fichier CSV d'essai.
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
    
    indices = {}
    for key, col_name in ESSAI_COLUMNS.items():
        try:
            indices[key] = header.index(col_name)
        except ValueError:
            raise ValueError(
                f"Colonne '{col_name}' non trouvée dans {filepath}. "
                f"Colonnes disponibles: {header}"
            )
    
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    
    return {
        'time': data[:, indices['time']],
        'M': data[:, indices['M']],
        'A': data[:, indices['A']],
        'Ti': data[:, indices['Ti']],
        'Ts': data[:, indices['Ts']],
        'filename': os.path.basename(filepath)
    }


def get_test_files(essais_dir=None):
    """
    Liste tous les fichiers CSV d'essai.
    """
    if essais_dir is None:
        essais_dir = ESSAIS_DIR
    
    if not os.path.exists(essais_dir):
        return []
    
    files = [os.path.join(essais_dir, f) for f in os.listdir(essais_dir) 
             if f.endswith('.csv')]
    return sorted(files)


# ============================================================================
# 4. CALCUL DES CONDITIONS LIMITES
# ============================================================================

def compute_T_inf(Ti):
    """
    Calcule la température d'échange T_inf à partir de Ti.
    """
    T_inf = Ti
    return T_inf


def create_bc_left_function(test_data, h_table, epsilon):
    """
    Crée la fonction de condition limite gauche.
    """
    t_essai = test_data['time']
    M_essai = test_data['M']
    A_essai = test_data['A']
    Ti_essai = test_data['Ti']
    Ts_essai = test_data['Ts']
    
    def bc_left(t):
        M = float(np.interp(t, t_essai, M_essai))
        A = float(np.interp(t, t_essai, A_essai))
        Ti = float(np.interp(t, t_essai, Ti_essai))
        Ts = float(np.interp(t, t_essai, Ts_essai))
        
        h = get_h_from_table(h_table, M, A)
        T_inf = compute_T_inf(Ti)
        
        return {
            'type': 'convection_radiation',
            'h': h,
            'T_inf': T_inf,
            'epsilon': epsilon,
            'T_s': Ts
        }
    
    return bc_left


def create_bc_right_adiabatic():
    """Crée la fonction de condition limite droite adiabatique."""
    def bc_right(t):
        return {'type': 'adiabatic'}
    return bc_right


# ============================================================================
# 5. EXÉCUTION DES SIMULATIONS
# ============================================================================

def run_simulation(empilement, test_data, h_table):
    """
    Exécute une simulation pour un empilement et un essai.
    """
    material_names = list(set(layer['material'] for layer in empilement['layers']))
    material_data = get_materials(*material_names)
    
    solver = ThermalSolver1D(empilement['layers'], material_data, Nx=empilement['Nx'])
    
    t_end = test_data['time'][-1]
    dt = empilement['dt']
    epsilon = empilement['epsilon']
    
    bc_left = create_bc_left_function(test_data, h_table, epsilon)
    bc_right = create_bc_right_adiabatic()
    
    n_steps = int(t_end / dt)
    save_every = max(1, n_steps // N_SAVE_POINTS)
    
    result = solver.solve(
        empilement['T_init'],
        t_end,
        dt,
        bc_left,
        bc_right,
        save_every=save_every,
        verbose=False
    )
    
    # Calcul des paramètres pour chaque instant sauvegardé
    t_essai = test_data['time']
    M_essai = test_data['M']
    A_essai = test_data['A']
    Ti_essai = test_data['Ti']
    Ts_essai = test_data['Ts']
    
    h_history = []
    T_inf_history = []
    Ts_history = []
    M_history = []
    A_history = []
    Ti_history = []
    
    for t in result['time']:
        M = float(np.interp(t, t_essai, M_essai))
        A = float(np.interp(t, t_essai, A_essai))
        Ti = float(np.interp(t, t_essai, Ti_essai))
        Ts = float(np.interp(t, t_essai, Ts_essai))
        
        h_history.append(get_h_from_table(h_table, M, A))
        T_inf_history.append(compute_T_inf(Ti))
        Ts_history.append(Ts)
        M_history.append(M)
        A_history.append(A)
        Ti_history.append(Ti)
    
    result['h'] = np.array(h_history)
    result['T_inf'] = np.array(T_inf_history)
    result['T_s'] = np.array(Ts_history)
    result['M'] = np.array(M_history)
    result['A'] = np.array(A_history)
    result['Ti'] = np.array(Ti_history)
    result['empilement'] = empilement
    result['test_name'] = test_data['filename']
    result['solver'] = solver
    
    return result


def _run_single_simulation_wrapper(args):
    """
    [OPTIM 5] Wrapper pour exécuter une simulation dans un processus parallèle.
    
    Nécessaire car multiprocessing.Pool.map ne peut passer qu'un seul argument.
    """
    zone_name, emp, test_file, h_table = args
    test_data = load_test_csv(test_file)
    result = run_simulation(emp, test_data, h_table)
    result['zone_name'] = zone_name
    return zone_name, result


def run_all_simulations(zone_empilements, h_tables, test_files, verbose=True, parallel=True):
    """
    [OPTIM 5] Exécute toutes les combinaisons zone × essai - VERSION PARALLÉLISÉE.
    
    Args:
        zone_empilements: Dict {zone_name: empilement}
        h_tables: Dict {zone_name: h_table}
        test_files: Liste des fichiers d'essai
        verbose: Afficher la progression
        parallel: Activer la parallélisation (True par défaut)
    
    Returns:
        Dict {zone_name: [results]}
    """
    all_results = {zone_name: [] for zone_name in zone_empilements}
    
    n_zones = len(zone_empilements)
    n_tests = len(test_files)
    n_total = n_zones * n_tests
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EXÉCUTION DES SIMULATIONS")
        print(f"{'='*70}")
        print(f"  {n_zones} zones × {n_tests} essais = {n_total} simulations")
        if parallel:
            print(f"  [OPTIM 5] Mode parallèle activé ({N_WORKERS} workers)")  # [OPTIM 5]
        print()
    
    # [OPTIM 5] Préparer la liste des tâches à exécuter
    tasks = []
    for zone_name, emp in zone_empilements.items():
        h_table = h_tables[zone_name]
        for test_file in test_files:
            tasks.append((zone_name, emp, test_file, h_table))
    
    if parallel and n_total > 1:
        # [OPTIM 5] Exécution parallèle avec Pool de processus
        if verbose:
            print(f"  Lancement de {n_total} simulations en parallèle...")
        
        with Pool(processes=N_WORKERS) as pool:  # [OPTIM 5] Pool de workers
            results_list = pool.map(_run_single_simulation_wrapper, tasks)  # [OPTIM 5] Map parallèle
        
        # [OPTIM 5] Réorganiser les résultats par zone
        for zone_name, result in results_list:
            all_results[zone_name].append(result)
        
        if verbose:
            count = 0
            for zone_name in zone_empilements:
                emp = zone_empilements[zone_name]
                print(f"\n▶ {zone_name} -> {emp['name']}: {emp['description']}")
                for result in all_results[zone_name]:
                    count += 1
                    T = result['T']
                    print(f"  [{count:2d}/{n_total}] {result['test_name']}... T: {T.min():.0f} - {T.max():.0f} K")
    else:
        # Mode séquentiel (pour debug ou si peu de simulations)
        count = 0
        for zone_name, emp in zone_empilements.items():
            h_table = h_tables[zone_name]
            
            if verbose:
                print(f"\n▶ {zone_name} -> {emp['name']}: {emp['description']}")
            
            for test_file in test_files:
                count += 1
                test_data = load_test_csv(test_file)
                
                if verbose:
                    print(f"  [{count:2d}/{n_total}] {test_data['filename']}...", end=" ", flush=True)
                
                result = run_simulation(emp, test_data, h_table)
                result['zone_name'] = zone_name
                all_results[zone_name].append(result)
                
                if verbose:
                    T = result['T']
                    print(f"T: {T.min():.0f} - {T.max():.0f} K")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ Toutes les simulations terminées")
        print(f"{'='*70}")
    
    return all_results


# ============================================================================
# 6. EXPORT DES RÉSULTATS
# ============================================================================

def export_results(all_results, output_dir):
    """
    Exporte les résultats en fichiers CSV.
    
    Args:
        all_results: Dict des résultats
        output_dir: Dossier de sortie (result/)
    """
    for zone_name, results in all_results.items():
        for result in results:
            test_name = result['test_name'].replace('.csv', '')
            emp_name = result['empilement']['name']
            
            filename = f"{zone_name}_{emp_name}_{test_name}.csv"
            filepath = os.path.join(output_dir, filename)
            
            data = np.column_stack([
                result['time'],
                result['T'][:, 0],
                result['T'][:, -1],
                result['h'],
                result['T_inf'],
                result['T_s'],
                result['M'],
                result['A'],
                result['Ti']
            ])
            np.savetxt(filepath, data, delimiter=',',
                      header='time_s,T_left_K,T_right_K,h_Wm2K,T_inf_K,Ts_K,M,A,Ti', 
                      comments='')


def print_summary(all_results):
    """Affiche un résumé tabulaire des résultats."""
    
    print("\n" + "="*105)
    print("RÉSUMÉ DES SIMULATIONS")
    print("="*105)
    
    for zone_name, results in all_results.items():
        emp = results[0]['empilement']
        print(f"\n▶ {zone_name} -> {emp['name']}: {emp['description']}")
        print("-"*85)
        print(f"  {'Essai':<28} {'T_init':>7} {'T_g_fin':>9} {'T_d_fin':>9} {'h_max':>7} {'Ti_max':>7} {'Ts_max':>7}")
        print("-"*85)
        
        for result in results:
            T_final = result['T'][-1, :]
            T_init = emp['T_init']
            h_max = result['h'].max()
            Ti_max = result['Ti'].max()
            Ts_max = result['T_s'].max()
            
            print(f"  {result['test_name']:<28} {T_init:>7.0f} "
                  f"{T_final[0]:>9.1f} {T_final[-1]:>9.1f} "
                  f"{h_max:>7.0f} {Ti_max:>7.0f} {Ts_max:>7.0f}")
    
    print("\n" + "="*105)


# ============================================================================
# 7. ANALYSE DES TEMPÉRATURES PAR MATÉRIAU
# ============================================================================

def get_material_indices(result):
    """
    Détermine les indices des nœuds correspondant à chaque matériau.
    
    Args:
        result: Résultat d'une simulation
    
    Returns:
        Dict {material_name: (idx_start, idx_end)}
    """
    empilement = result['empilement']
    layers = empilement['layers']
    
    x = result['x']
    Nx = len(x)
    
    # Calculer les positions des interfaces
    interface_positions = [0.0]
    for layer in layers:
        interface_positions.append(interface_positions[-1] + layer['thickness'])
    
    # Trouver les indices pour chaque couche/matériau
    material_indices = {}
    for i, layer in enumerate(layers):
        x_start = interface_positions[i]
        x_end = interface_positions[i + 1]
        
        # Indices des nœuds dans cette couche
        idx_start = np.searchsorted(x, x_start)
        idx_end = np.searchsorted(x, x_end, side='right') - 1
        
        # S'assurer d'avoir au moins un nœud
        if idx_end < idx_start:
            idx_end = idx_start
        if idx_end >= Nx:
            idx_end = Nx - 1
        
        material_name = layer['material']
        
        # Si le matériau existe déjà, étendre la plage
        if material_name in material_indices:
            existing = material_indices[material_name]
            material_indices[material_name] = (
                min(existing[0], idx_start),
                max(existing[1], idx_end)
            )
        else:
            material_indices[material_name] = (idx_start, idx_end)
    
    return material_indices


def compute_material_mean_temperatures(result):
    """
    Calcule la température moyenne temporelle de chaque matériau.
    
    Pour chaque matériau:
    1. Moyenne spatiale sur les nœuds du matériau à chaque instant
    2. Moyenne temporelle sur toute la simulation
    
    Args:
        result: Résultat d'une simulation
    
    Returns:
        Dict {material_name: T_mean}
    """
    T = result['T']  # Shape: (n_times, n_nodes)
    material_indices = get_material_indices(result)
    
    mean_temps = {}
    for material_name, (idx_start, idx_end) in material_indices.items():
        # Température dans ce matériau: moyenne spatiale puis temporelle
        T_material = T[:, idx_start:idx_end+1]
        
        # Moyenne spatiale à chaque instant
        T_spatial_mean = np.mean(T_material, axis=1)
        
        # Moyenne temporelle
        T_mean = np.mean(T_spatial_mean)
        
        mean_temps[material_name] = T_mean
    
    return mean_temps


def compute_material_max_temperatures(result):
    """
    Calcule la température maximale atteinte dans chaque matériau.
    
    Args:
        result: Résultat d'une simulation
    
    Returns:
        Dict {material_name: T_max}
    """
    T = result['T']
    material_indices = get_material_indices(result)
    
    max_temps = {}
    for material_name, (idx_start, idx_end) in material_indices.items():
        T_material = T[:, idx_start:idx_end+1]
        max_temps[material_name] = np.max(T_material)
    
    return max_temps


def analyze_essais_by_material(all_results, metric='mean'):
    """
    Analyse et classe les essais par température pour chaque matériau.
    
    Args:
        all_results: Dict des résultats {zone_name: [results]}
        metric: 'mean' pour température moyenne, 'max' pour maximum
    
    Returns:
        Dict structuré:
        {
            zone_name: {
                'empilement': empilement,
                'materials': [liste des matériaux dans l'ordre],
                'essais': [
                    {
                        'name': 'essai_01.csv',
                        'temperatures': {material: T_value, ...},
                        'T_global_mean': valeur
                    },
                    ...
                ]
            }
        }
    """
    if metric == 'mean':
        compute_func = compute_material_mean_temperatures
    else:
        compute_func = compute_material_max_temperatures
    
    analysis = {}
    
    for zone_name, results in all_results.items():
        emp = results[0]['empilement']
        
        # Liste unique des matériaux dans l'ordre d'apparition
        materials = []
        for layer in emp['layers']:
            if layer['material'] not in materials:
                materials.append(layer['material'])
        
        essais_data = []
        for result in results:
            temps = compute_func(result)
            
            # Température globale moyenne de l'empilement
            T_global = np.mean(list(temps.values()))
            
            essais_data.append({
                'name': result['test_name'].replace('.csv', ''),
                'temperatures': temps,
                'T_global_mean': T_global
            })
        
        analysis[zone_name] = {
            'empilement': emp,
            'materials': materials,
            'essais': essais_data
        }
    
    return analysis


def rank_essais_by_material(analysis, n_top=None):
    """
    Classe les essais par température pour chaque matériau.
    
    Args:
        analysis: Résultat de analyze_essais_by_material
        n_top: Nombre d'essais à retourner (None = tous)
    
    Returns:
        Dict {zone_name: {material: [(essai_name, T_value, rank), ...]}}
    """
    rankings = {}
    
    for zone_name, zone_data in analysis.items():
        materials = zone_data['materials']
        essais = zone_data['essais']
        
        zone_rankings = {}
        
        for material in materials:
            # Trier par température décroissante
            sorted_essais = sorted(
                essais,
                key=lambda e: e['temperatures'].get(material, 0),
                reverse=True
            )
            
            # Limiter au top N
            if n_top is not None:
                sorted_essais = sorted_essais[:n_top]
            
            # Créer la liste avec rang
            ranked = [
                (e['name'], e['temperatures'].get(material, 0), i+1)
                for i, e in enumerate(sorted_essais)
            ]
            
            zone_rankings[material] = ranked
        
        # Ajouter le classement global
        sorted_global = sorted(essais, key=lambda e: e['T_global_mean'], reverse=True)
        if n_top is not None:
            sorted_global = sorted_global[:n_top]
        zone_rankings['GLOBAL'] = [
            (e['name'], e['T_global_mean'], i+1)
            for i, e in enumerate(sorted_global)
        ]
        
        rankings[zone_name] = zone_rankings
    
    return rankings


def print_material_ranking(analysis, rankings, n_top=None):
    """
    Affiche le classement des essais par matériau.
    """
    from param_simu import N_TOP_ESSAIS
    
    if n_top is None:
        n_top = N_TOP_ESSAIS
    
    print("\n" + "="*100)
    print("CLASSEMENT DES ESSAIS PAR TEMPÉRATURE MOYENNE DES MATÉRIAUX")
    print("="*100)
    
    for zone_name, zone_data in analysis.items():
        emp = zone_data['empilement']
        materials = zone_data['materials']
        zone_rank = rankings[zone_name]
        
        print(f"\n{'─'*100}")
        print(f"▶ {zone_name} -> {emp['name']}: {emp['description']}")
        print(f"{'─'*100}")
        
        # En-tête
        header = f"  {'Rang':<6}"
        for mat in materials:
            header += f" {mat[:15]:<18}"
        header += f" {'GLOBAL':<18}"
        print(header)
        print("  " + "-"*94)
        
        # Lignes
        for i in range(min(n_top, len(zone_data['essais']))):
            line = f"  {i+1:<6}"
            
            for mat in materials:
                if i < len(zone_rank[mat]):
                    name, temp, _ = zone_rank[mat][i]
                    line += f" {name[:12]:<12} {temp:>5.0f}K"
                else:
                    line += " " * 18
            
            # Global
            if i < len(zone_rank['GLOBAL']):
                name, temp, _ = zone_rank['GLOBAL'][i]
                line += f" {name[:12]:<12} {temp:>5.0f}K"
            
            print(line)
    
    print("\n" + "="*100)
