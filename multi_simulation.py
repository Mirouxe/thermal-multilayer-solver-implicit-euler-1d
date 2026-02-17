"""
Script principal de simulation thermique multicouche
====================================================

Ce script orchestre les simulations et crée un dossier de sortie structuré:

    results/simu_YYYYMMDD_HHMMSS/
        ├── result/      # Résultats CSV (T, h, M, A, Ti, Ts, ...)
        ├── visu/        # Visualisations PNG
        └── calcul/      # Copie des scripts (pour relancer la simulation)

Modules utilisés:
- param_simu.py: Paramètres de configuration
- empilement_library.py: Définitions des empilements
- utils_simulation.py: Fonctions de simulation
- post.py: Visualisation des résultats

OPTIMISATIONS IMPLÉMENTÉES:
--------------------------
[OPTIM 1] Solveur tridiagonal (solver.py) - Résolution O(n) au lieu de O(n³)
[OPTIM 2] Vectorisation get_properties_at_nodes (solver.py) - Interp par groupes matériaux
[OPTIM 3] Vectorisation nœuds intérieurs (solver.py) - Construction matrice sans boucle
[OPTIM 4] Pré-calcul indices matériaux (solver.py) - Évite lookup répétés
[OPTIM 5] Parallélisation simulations (utils_simulation.py) - multiprocessing.Pool

Paramètres d'optimisation modifiables dans param_simu.py:
- PARALLEL_ENABLED: True/False pour activer/désactiver la parallélisation

Usage:
    python multi_simulation.py
"""

from datetime import datetime
import os

from param_simu import (
    TABLES_DIR, ESSAIS_DIR, ZONE_TO_EMPILEMENT, TABLE_COLUMNS, N_TOP_ESSAIS,
    PARALLEL_ENABLED  # [OPTIM 5] Paramètre de parallélisation
)
from utils_simulation import (
    create_output_directory,
    copy_scripts_to_calcul,
    load_all_h_tables,
    get_test_files,
    load_test_csv,
    run_all_simulations,
    export_results,
    print_summary,
    analyze_essais_by_material,
    rank_essais_by_material,
    print_material_ranking
)
from post import generate_all_plots


def main():
    """Fonction principale."""
    
    print("\n" + "="*70)
    print("SIMULATION THERMIQUE MULTICOUCHE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. Créer le dossier de sortie
    # -------------------------------------------------------------------------
    print("\n1. Création du dossier de sortie...")
    output_dirs = create_output_directory()
    print(f"   Dossier: {output_dirs['root']}")
    print(f"   ├── result/  (résultats CSV)")
    print(f"   ├── visu/    (visualisations)")
    print(f"   └── calcul/  (scripts)")
    
    # -------------------------------------------------------------------------
    # 2. Afficher la configuration
    # -------------------------------------------------------------------------
    print("\n2. Configuration:")
    print(f"   Colonnes tables: M='{TABLE_COLUMNS['M']}', A='{TABLE_COLUMNS['A']}', h='{TABLE_COLUMNS['h']}'")
    print(f"   Mapping zones -> empilements:")
    for zone, emp in ZONE_TO_EMPILEMENT.items():
        print(f"      {zone} -> {emp}")
    
    # -------------------------------------------------------------------------
    # 3. Charger les tables h(M, A)
    # -------------------------------------------------------------------------
    print(f"\n3. Chargement des tables h(M, A) depuis '{TABLES_DIR}/'...")
    h_tables, zone_empilements = load_all_h_tables(TABLES_DIR)
    
    for zone_name, table in h_tables.items():
        print(f"      M ∈ [{table['M_range'][0]:.1f}, {table['M_range'][1]:.1f}], "
              f"A ∈ [{table['A_range'][0]:.1f}, {table['A_range'][1]:.1f}], "
              f"h ∈ [{table['h'].min():.0f}, {table['h'].max():.0f}] W/m²K")
    
    # -------------------------------------------------------------------------
    # 4. Charger les fichiers d'essai
    # -------------------------------------------------------------------------
    print(f"\n4. Fichiers d'essai depuis '{ESSAIS_DIR}/':")
    test_files = get_test_files(ESSAIS_DIR)
    
    if not test_files:
        raise FileNotFoundError(f"Aucun fichier d'essai trouvé dans {ESSAIS_DIR}/")
    
    for f in test_files:
        test_data = load_test_csv(f)
        print(f"   • {os.path.basename(f)}: "
              f"t=[{test_data['time'][0]:.0f}, {test_data['time'][-1]:.0f}]s, "
              f"Ti=[{test_data['Ti'].min():.0f}, {test_data['Ti'].max():.0f}]K")
    
    # -------------------------------------------------------------------------
    # 5. Exécuter les simulations
    # -------------------------------------------------------------------------
    print("\n5. Exécution des simulations...")
    # [OPTIM 5] Utilisation de la parallélisation si activée dans param_simu.py
    all_results = run_all_simulations(
        zone_empilements, h_tables, test_files, 
        verbose=True, 
        parallel=PARALLEL_ENABLED  # [OPTIM 5] Contrôle via param_simu.PARALLEL_ENABLED
    )
    
    # -------------------------------------------------------------------------
    # 6. Afficher le résumé
    # -------------------------------------------------------------------------
    print_summary(all_results)
    
    # -------------------------------------------------------------------------
    # 6b. Analyse et classement par température matériau
    # -------------------------------------------------------------------------
    print("\n6b. Analyse des températures par matériau...")
    analysis = analyze_essais_by_material(all_results, metric='mean')
    rankings = rank_essais_by_material(analysis, n_top=N_TOP_ESSAIS)
    print_material_ranking(analysis, rankings, n_top=N_TOP_ESSAIS)
    
    # -------------------------------------------------------------------------
    # 7. Exporter les résultats CSV
    # -------------------------------------------------------------------------
    print("\n6. Export des résultats CSV...")
    export_results(all_results, output_dirs['result'])
    print(f"   Résultats exportés dans: {output_dirs['result']}")
    
    # -------------------------------------------------------------------------
    # 8. Générer les visualisations
    # -------------------------------------------------------------------------
    print("\n7. Génération des visualisations...")
    generate_all_plots(all_results, h_tables, zone_empilements, test_files,
                       output_dir=output_dirs['visu'], save=True, show=False,
                       n_top_ranking=N_TOP_ESSAIS)
    
    # -------------------------------------------------------------------------
    # 9. Copier les scripts dans calcul/
    # -------------------------------------------------------------------------
    print("\n8. Archivage des scripts...")
    copy_scripts_to_calcul(output_dirs['calcul'])
    print(f"   Scripts copiés dans: {output_dirs['calcul']}")
    
    # -------------------------------------------------------------------------
    # Résumé final
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("SIMULATION TERMINÉE")
    print("="*70)
    print(f"\nDossier de sortie: {output_dirs['root']}")
    print(f"  • result/  : Fichiers CSV avec T, h, M, A, Ti, Ts")
    print(f"  • visu/    : Graphiques PNG + matrices de classement")
    print(f"  • calcul/  : Scripts pour relancer cette simulation")
    print(f"\nClassement: Top {N_TOP_ESSAIS} essais par matériau")
    print(f"  (Modifier N_TOP_ESSAIS dans param_simu.py pour ajuster)")
    print(f"\nOptimisations actives:")
    print(f"  • [OPTIM 1] Solveur tridiagonal O(n)")      # [OPTIM 1]
    print(f"  • [OPTIM 2] Vectorisation propriétés")       # [OPTIM 2]
    print(f"  • [OPTIM 3] Vectorisation nœuds intérieurs") # [OPTIM 3]
    print(f"  • [OPTIM 4] Pré-calcul indices matériaux")   # [OPTIM 4]
    print(f"  • [OPTIM 5] Parallélisation: {'OUI' if PARALLEL_ENABLED else 'NON'}")  # [OPTIM 5]
    print(f"  (Modifier PARALLEL_ENABLED dans param_simu.py)")
    print("\nPour relancer cette simulation:")
    print(f"  cd {output_dirs['calcul']}")
    print(f"  python multi_simulation.py")
    print("="*70)
    
    return all_results, h_tables, zone_empilements, test_files, output_dirs


if __name__ == '__main__':
    results, h_tables, zone_empilements, test_files, output_dirs = main()
