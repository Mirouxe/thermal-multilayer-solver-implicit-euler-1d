"""
Bibliothèque des empilements
============================

Ce fichier définit tous les empilements de matériaux disponibles.

Chaque empilement contient:
- name: Nom unique de l'empilement
- description: Description textuelle
- layers: Liste des couches [{'material': str, 'thickness': float}, ...]
- table_file: Fichier CSV de la table h(M, A)
- Nx: Nombre de nœuds spatiaux
- color: Couleur pour les graphiques
- T_init: Température initiale [K]
- dt: Pas de temps [s]
- epsilon: Émissivité pour le rayonnement
"""


# ============================================================================
# LISTE DES EMPILEMENTS
# ============================================================================

EMPILEMENTS = [
    # -------------------------------------------------------------------------
    # Empilement 1: Acier seul
    # -------------------------------------------------------------------------
    {
        'name': 'Acier_10mm',
        'description': 'Plaque acier inox 304 - 10 mm',
        'layers': [
            {'material': 'steel_304', 'thickness': 0.010}
        ],
        'Nx': 51,
        'color': 'blue',
        'T_init': 300.0,
        'dt': 0.2,
        'epsilon': 0.3,
    },
    
    # -------------------------------------------------------------------------
    # Empilement 2: Acier + Isolant (laine de verre)
    # -------------------------------------------------------------------------
    {
        'name': 'Acier_Isolant',
        'description': 'Acier 5mm + Laine de verre 30mm',
        'layers': [
            {'material': 'steel_304', 'thickness': 0.005},
            {'material': 'glass_wool', 'thickness': 0.030}
        ],
        'Nx': 101,
        'color': 'green',
        'T_init': 300.0,
        'dt': 0.3,
        'epsilon': 0.4,
    },
    
    # -------------------------------------------------------------------------
    # Empilement 3: Multicouche (métal-isolant-métal)
    # -------------------------------------------------------------------------
    {
        'name': 'Multicouche',
        'description': 'Acier 5mm + Céramique 15mm + Alu 5mm',
        'layers': [
            {'material': 'steel_304', 'thickness': 0.005},
            {'material': 'ceramic_fiber', 'thickness': 0.015},
            {'material': 'aluminum_6061', 'thickness': 0.005}
        ],
        'Nx': 101,
        'color': 'red',
        'T_init': 300.0,
        'dt': 0.2,
        'epsilon': 0.35,
    },
]


# ============================================================================
# FONCTIONS D'ACCÈS
# ============================================================================

def get_empilement(name):
    """
    Récupère un empilement par son nom.
    
    Args:
        name: Nom de l'empilement
    
    Returns:
        Dict de l'empilement ou None si non trouvé
    """
    for emp in EMPILEMENTS:
        if emp['name'] == name:
            return emp
    return None


def get_all_empilements():
    """
    Retourne la liste de tous les empilements.
    
    Returns:
        Liste des empilements
    """
    return EMPILEMENTS


def list_empilements():
    """
    Affiche la liste des empilements disponibles.
    """
    print("\nEmpilements disponibles:")
    print("-" * 60)
    for emp in EMPILEMENTS:
        layers_str = ' + '.join([
            f"{l['material']}({l['thickness']*1000:.0f}mm)" 
            for l in emp['layers']
        ])
        print(f"  • {emp['name']}: {layers_str}")
    print("-" * 60)


def add_empilement(empilement):
    """
    Ajoute un nouvel empilement à la bibliothèque.
    
    Args:
        empilement: Dict définissant l'empilement
    """
    # Vérifier les champs requis
    required_fields = ['name', 'layers', 'table_file', 'Nx', 'T_init', 'dt', 'epsilon']
    for field in required_fields:
        if field not in empilement:
            raise ValueError(f"Champ requis manquant: {field}")
    
    # Vérifier que le nom n'existe pas déjà
    if get_empilement(empilement['name']) is not None:
        raise ValueError(f"Empilement '{empilement['name']}' existe déjà")
    
    EMPILEMENTS.append(empilement)
    print(f"Empilement '{empilement['name']}' ajouté")


def create_empilement(name, layers, table_file, Nx=101, T_init=300.0, dt=0.2, 
                      epsilon=0.3, color='black', description=''):
    """
    Crée un nouvel empilement avec les paramètres donnés.
    
    Args:
        name: Nom unique
        layers: Liste des couches
        table_file: Fichier de la table h(M, A)
        Nx: Nombre de nœuds
        T_init: Température initiale [K]
        dt: Pas de temps [s]
        epsilon: Émissivité
        color: Couleur pour les graphiques
        description: Description textuelle
    
    Returns:
        Dict de l'empilement
    """
    return {
        'name': name,
        'description': description if description else name,
        'layers': layers,
        'table_file': table_file,
        'Nx': Nx,
        'color': color,
        'T_init': T_init,
        'dt': dt,
        'epsilon': epsilon,
    }
