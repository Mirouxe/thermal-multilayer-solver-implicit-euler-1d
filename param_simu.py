"""
Paramètres statiques de simulation
==================================

Ce fichier contient tous les paramètres de configuration:
- Chemins des dossiers
- Mapping des noms de colonnes
- Mapping des zones vers les empilements
- Paramètres par défaut
"""

# ============================================================================
# CHEMINS DES DOSSIERS
# ============================================================================

TABLES_DIR = 'tables'       # Dossier des tables h(M, A)
ESSAIS_DIR = 'essais'       # Dossier des fichiers d'essai
RESULTS_DIR = 'results'     # Dossier d'export des résultats

# ============================================================================
# NOMS DES COLONNES DANS LES FICHIERS CSV
# ============================================================================

# Noms des colonnes dans les tables h(M, A)
# Format: {clé_interne: nom_dans_csv}
TABLE_COLUMNS = {
    'M': 'Mach',      # Colonne pour le paramètre M
    'A': 'Alph',      # Colonne pour le paramètre A
    'h': 'Ste',       # Colonne pour le coefficient h
}

# Noms des colonnes dans les fichiers d'essai
# Format: {clé_interne: nom_dans_csv}
ESSAI_COLUMNS = {
    'time': 'time',   # Colonne temps
    'M': 'M',         # Colonne paramètre M
    'A': 'A',         # Colonne paramètre A
    'Ti': 'Ti',       # Colonne température Ti
    'Ts': 'Ts',       # Colonne température Ts
}

# ============================================================================
# MAPPING ZONES -> EMPILEMENTS
# ============================================================================

# Chaque fichier de table est nommé: data_<nom_zone>.csv
# Ce dictionnaire associe chaque zone à un empilement
# Format: {nom_zone: nom_empilement}
ZONE_TO_EMPILEMENT = {
    'Zone1': 'Acier_10mm',
    'Zone2': 'Acier_Isolant',
    'Zone3': 'Multicouche',
}

# Préfixe des fichiers de table
TABLE_FILE_PREFIX = 'data_'

# ============================================================================
# PARAMÈTRES PAR DÉFAUT
# ============================================================================

# Nombre d'instants à sauvegarder pendant la simulation
N_SAVE_POINTS = 100

# ============================================================================
# PARAMÈTRES D'OPTIMISATION
# ============================================================================

# [OPTIM 5] Activer la parallélisation des simulations (True = parallèle, False = séquentiel)
PARALLEL_ENABLED = True

# ============================================================================
# PARAMÈTRES DE VISUALISATION
# ============================================================================

FIGSIZE_SMALL = (10, 6)
FIGSIZE_MEDIUM = (14, 8)
FIGSIZE_LARGE = (16, 12)
DPI = 150

# ============================================================================
# PARAMÈTRES DE CLASSEMENT DES ESSAIS
# ============================================================================

# Nombre d'essais à afficher dans le classement (Top N)
N_TOP_ESSAIS = 10
