"""
Post-traitement et visualisation
================================

Ce fichier contient toutes les fonctions de visualisation:
- Affichage des tables h(M, A)
- Affichage des conditions d'essai
- Graphiques des résultats par zone/empilement
- Comparaison globale
- Matrices de classement des essais par température matériau
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from param_simu import FIGSIZE_SMALL, FIGSIZE_MEDIUM, FIGSIZE_LARGE, DPI, N_TOP_ESSAIS
from utils_simulation import (
    load_test_csv, compute_T_inf,
    analyze_essais_by_material, rank_essais_by_material
)


# ============================================================================
# 1. VISUALISATION DES TABLES h(M, A)
# ============================================================================

def plot_h_table(h_table, name, ax=None):
    """
    Visualise une table h(M, A) sur un axe.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    
    M = h_table['M']
    A = h_table['A']
    h = h_table['h']
    
    sc = ax.scatter(A, M, c=h, cmap='viridis', s=100, edgecolors='black')
    
    for i in range(len(M)):
        ax.annotate(f'{h[i]:.0f}', (A[i], M[i]), 
                   textcoords="offset points", xytext=(0, 5),
                   ha='center', fontsize=7)
    
    ax.set_xlabel('A')
    ax.set_ylabel('M')
    ax.set_title(f'{name}\nh(M, A) [W/m²K]')
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('h [W/m²K]')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_h_tables(h_tables, zone_empilements, figsize=None):
    """
    Visualise toutes les tables h(M, A).
    """
    n_tables = len(h_tables)
    
    if figsize is None:
        figsize = (5 * n_tables, 4)
    
    fig, axes = plt.subplots(1, n_tables, figsize=figsize)
    
    if n_tables == 1:
        axes = [axes]
    
    for ax, (zone_name, table) in zip(axes, h_tables.items()):
        emp_name = zone_empilements[zone_name]['name']
        title = f"{zone_name}\n({emp_name})"
        plot_h_table(table, title, ax)
    
    plt.suptitle('Tables d\'interpolation h(M, A) par zone', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# 2. VISUALISATION DES CONDITIONS D'ESSAI
# ============================================================================

def plot_test_conditions(test_files, figsize=None):
    """
    Trace les conditions d'essai (M, A, Ti, Ts).
    """
    if figsize is None:
        figsize = FIGSIZE_LARGE
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_files)))
    
    for test_file, color in zip(test_files, colors):
        test_data = load_test_csv(test_file)
        t = test_data['time']
        label = test_data['filename'].replace('.csv', '')
        
        axes[0, 0].plot(t, test_data['M'], '-', color=color, linewidth=2, label=label)
        axes[0, 1].plot(t, test_data['A'], '-', color=color, linewidth=2, label=label)
        axes[0, 2].plot(t, test_data['Ti'], '-', color=color, linewidth=2, label=label)
        axes[1, 0].plot(t, test_data['Ts'], '-', color=color, linewidth=2, label=label)
        
        T_inf = [compute_T_inf(ti) for ti in test_data['Ti']]
        axes[1, 1].plot(t, T_inf, '-', color=color, linewidth=2, label=label)
        
        dT = test_data['Ti'] - test_data['Ts']
        axes[1, 2].plot(t, dT, '-', color=color, linewidth=2, label=label)
    
    titles = ['Paramètre M(t)', 'Paramètre A(t)', 'Paramètre Ti(t)',
              'Température Ts(t)', 'T_inf = f(Ti)', 'Ti - Ts']
    ylabels = ['M', 'A', 'Ti [K]', 'Ts [K]', 'T_inf [K]', 'ΔT [K]']
    
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_xlabel('Temps [s]')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Conditions d\'essai', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# 3. VISUALISATION DES RÉSULTATS PAR ZONE
# ============================================================================

def plot_zone_results(zone_name, results, figsize=None):
    """
    Trace les résultats d'une zone.
    """
    if figsize is None:
        figsize = FIGSIZE_LARGE
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for result, color in zip(results, colors):
        time = result['time']
        T = result['T']
        label = result['test_name'].replace('.csv', '')
        
        axes[0, 0].plot(time, T[:, 0], '-', color=color, linewidth=2, label=label)
        axes[0, 1].plot(time, T[:, -1], '-', color=color, linewidth=2, label=label)
        axes[0, 2].plot(time, result['h'], '-', color=color, linewidth=2, label=label)
        
        axes[1, 0].plot(time, result['T_inf'], '-', color=color, linewidth=2, label=label)
        axes[1, 0].plot(time, result['T_s'], '--', color=color, linewidth=1, alpha=0.7)
        
        x = result['x'] * 1000
        axes[1, 1].plot(x, T[-1, :], '-', color=color, linewidth=2, label=label)
        
        dT = T[:, 0] - T[:, -1]
        axes[1, 2].plot(time, dT, '-', color=color, linewidth=2, label=label)
    
    titles = ['Surface gauche T(t)', 'Surface droite T(t)', 'Coefficient h(t)',
              'T_inf (—) et Ts (--)', 'Profils finaux T(x)', 'Gradient ΔT']
    ylabels = ['T [K]', 'T [K]', 'h [W/m²K]', 'T [K]', 'T [K]', 'ΔT [K]']
    xlabels = ['Temps [s]', 'Temps [s]', 'Temps [s]', 
               'Temps [s]', 'Position [mm]', 'Temps [s]']
    
    for ax, title, ylabel, xlabel in zip(axes.flat, titles, ylabels, xlabels):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Interfaces
    emp = results[0]['empilement']
    cum_thick = np.cumsum([l['thickness'] for l in emp['layers']])
    for x_int in cum_thick[:-1]:
        axes[1, 1].axvline(x=x_int*1000, color='gray', linestyle='--', alpha=0.5)
    
    emp_name = emp['name']
    plt.suptitle(f"{zone_name} -> {emp_name}: {emp['description']}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# 4. COMPARAISON GLOBALE
# ============================================================================

def plot_comparison(all_results, zone_empilements, figsize=None):
    """
    Compare toutes les zones sur un même graphique.
    """
    if figsize is None:
        figsize = FIGSIZE_MEDIUM
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    zone_colors = {
        zone: emp['color'] 
        for zone, emp in zone_empilements.items()
    }
    linestyles = ['-', '--', '-.', ':']
    
    for zone_name, results in all_results.items():
        color = zone_colors.get(zone_name, 'black')
        
        for i, result in enumerate(results):
            ls = linestyles[i % len(linestyles)]
            label = f"{zone_name} - {result['test_name'].replace('.csv', '')}"
            
            axes[0].plot(result['time'], result['T'][:, 0], 
                        linestyle=ls, color=color, linewidth=1.5, label=label)
            axes[1].plot(result['time'], result['T'][:, -1],
                        linestyle=ls, color=color, linewidth=1.5, label=label)
    
    axes[0].set_xlabel('Temps [s]')
    axes[0].set_ylabel('T [K]')
    axes[0].set_title('Surface GAUCHE')
    axes[0].legend(fontsize=6, ncol=2, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('T [K]')
    axes[1].set_title('Surface DROITE')
    axes[1].legend(fontsize=6, ncol=2, loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Comparaison de toutes les zones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# 5. MATRICES DE CLASSEMENT DES ESSAIS PAR TEMPÉRATURE MATÉRIAU
# ============================================================================

def plot_material_ranking_matrix(zone_name, zone_data, rankings, n_top=None, 
                                  metric='mean', figsize=None):
    """
    Crée une matrice visuelle du classement des essais par matériau.
    
    La matrice montre:
    - Colonnes: Matériaux de l'empilement + colonne GLOBAL
    - Lignes: Top N essais (classés par température décroissante)
    - Couleurs: Intensité selon la température
    - Annotations: Nom de l'essai + température
    
    Args:
        zone_name: Nom de la zone
        zone_data: Données d'analyse pour cette zone
        rankings: Classement des essais
        n_top: Nombre d'essais à afficher
        metric: 'mean' ou 'max'
        figsize: Taille de la figure
    
    Returns:
        Figure matplotlib
    """
    if n_top is None:
        n_top = N_TOP_ESSAIS
    
    emp = zone_data['empilement']
    materials = zone_data['materials']
    zone_rank = rankings[zone_name]
    
    # Ajouter GLOBAL aux colonnes
    columns = materials + ['GLOBAL']
    n_cols = len(columns)
    n_rows = min(n_top, len(zone_data['essais']))
    
    if figsize is None:
        figsize = (3 + 2.5 * n_cols, 1 + 0.6 * n_rows)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Créer la matrice des températures
    temp_matrix = np.zeros((n_rows, n_cols))
    labels_matrix = [['' for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Remplir la matrice
    for j, col in enumerate(columns):
        ranked_list = zone_rank.get(col, [])
        for i in range(n_rows):
            if i < len(ranked_list):
                name, temp, rank = ranked_list[i]
                temp_matrix[i, j] = temp
                labels_matrix[i][j] = f"{name}\n{temp:.0f}K"
            else:
                temp_matrix[i, j] = np.nan
                labels_matrix[i][j] = '-'
    
    # Normaliser pour la colormap (par colonne pour mieux voir les variations)
    temp_normalized = np.zeros_like(temp_matrix)
    for j in range(n_cols):
        col_data = temp_matrix[:, j]
        valid = ~np.isnan(col_data)
        if np.any(valid):
            vmin, vmax = col_data[valid].min(), col_data[valid].max()
            if vmax > vmin:
                temp_normalized[:, j] = (col_data - vmin) / (vmax - vmin)
            else:
                temp_normalized[:, j] = 0.5
    
    # Créer le heatmap
    cmap = plt.cm.YlOrRd
    im = ax.imshow(temp_normalized, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Ajouter les annotations
    for i in range(n_rows):
        for j in range(n_cols):
            text = labels_matrix[i][j]
            # Couleur du texte selon l'intensité
            if temp_normalized[i, j] > 0.6:
                text_color = 'white'
            else:
                text_color = 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                   color=text_color, fontweight='bold')
    
    # Configuration des axes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(columns, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"#{i+1}" for i in range(n_rows)], fontsize=10)
    
    ax.set_xlabel('Matériau', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classement', fontsize=12, fontweight='bold')
    
    # Titre
    metric_label = "moyenne" if metric == 'mean' else "maximale"
    title = f"{zone_name} → {emp['name']}\nClassement par température {metric_label}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Grille
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Température relative\n(par colonne)', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_all_ranking_matrices(all_results, n_top=None, metric='mean'):
    """
    Génère les matrices de classement pour toutes les zones.
    
    Args:
        all_results: Résultats de toutes les simulations
        n_top: Nombre d'essais à afficher
        metric: 'mean' ou 'max'
    
    Returns:
        Dict {zone_name: figure}
    """
    if n_top is None:
        n_top = N_TOP_ESSAIS
    
    # Analyser les résultats
    analysis = analyze_essais_by_material(all_results, metric=metric)
    rankings = rank_essais_by_material(analysis, n_top=n_top)
    
    figures = {}
    for zone_name, zone_data in analysis.items():
        fig = plot_material_ranking_matrix(
            zone_name, zone_data, rankings, n_top=n_top, metric=metric
        )
        figures[zone_name] = fig
    
    return figures, analysis, rankings


def plot_global_ranking_summary(analysis, rankings, n_top=None, figsize=None):
    """
    Crée un résumé visuel comparant les classements globaux de toutes les zones.
    
    Args:
        analysis: Résultat de analyze_essais_by_material
        rankings: Résultat de rank_essais_by_material
        n_top: Nombre d'essais à afficher
        figsize: Taille de la figure
    
    Returns:
        Figure matplotlib
    """
    if n_top is None:
        n_top = N_TOP_ESSAIS
    
    n_zones = len(analysis)
    n_rows = min(n_top, max(len(z['essais']) for z in analysis.values()))
    
    if figsize is None:
        figsize = (4 * n_zones, 1.5 + 0.5 * n_rows)
    
    fig, axes = plt.subplots(1, n_zones, figsize=figsize)
    if n_zones == 1:
        axes = [axes]
    
    for ax, (zone_name, zone_data) in zip(axes, analysis.items()):
        emp = zone_data['empilement']
        global_rank = rankings[zone_name]['GLOBAL']
        
        # Données pour le bar chart horizontal
        essais = [r[0] for r in global_rank[:n_rows]]
        temps = [r[1] for r in global_rank[:n_rows]]
        
        # Couleurs selon la température
        norm_temps = np.array(temps)
        if norm_temps.max() > norm_temps.min():
            norm_temps = (norm_temps - norm_temps.min()) / (norm_temps.max() - norm_temps.min())
        else:
            norm_temps = np.ones_like(norm_temps) * 0.5
        colors = plt.cm.YlOrRd(norm_temps)
        
        # Bar chart horizontal
        y_pos = np.arange(len(essais))
        bars = ax.barh(y_pos, temps, color=colors, edgecolor='black', linewidth=0.5)
        
        # Annotations
        for i, (bar, temp) in enumerate(zip(bars, temps)):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   f'{temp:.0f}K', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(essais, fontsize=9)
        ax.invert_yaxis()  # Top = rank 1
        ax.set_xlabel('T moyenne globale [K]', fontsize=10)
        ax.set_title(f"{zone_name}\n{emp['name']}", fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Ajouter le rang
        for i, essai in enumerate(essais):
            ax.text(-0.02, i, f"#{i+1}", transform=ax.get_yaxis_transform(),
                   ha='right', va='center', fontsize=9, color='gray')
    
    plt.suptitle('Classement global des essais par zone', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# 6. GÉNÉRATION DE TOUS LES GRAPHIQUES
# ============================================================================

def generate_all_plots(all_results, h_tables, zone_empilements, test_files,
                       output_dir, save=True, show=False, n_top_ranking=None):
    """
    Génère et sauvegarde tous les graphiques.
    
    Args:
        all_results: Dict des résultats
        h_tables: Dict des tables h(M, A)
        zone_empilements: Dict {zone: empilement}
        test_files: Liste des fichiers d'essai
        output_dir: Dossier visu/ pour sauvegarder les images
        save: Sauvegarder les figures
        show: Afficher les figures
        n_top_ranking: Nombre d'essais dans les classements (None = N_TOP_ESSAIS)
    """
    if n_top_ranking is None:
        n_top_ranking = N_TOP_ESSAIS
    
    figures = {}
    
    # Tables h(M, A)
    fig = plot_h_tables(h_tables, zone_empilements)
    figures['h_tables'] = fig
    if save:
        filepath = os.path.join(output_dir, 'h_tables.png')
        fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"   - {filepath}")
    
    # Conditions d'essai
    fig = plot_test_conditions(test_files)
    figures['test_conditions'] = fig
    if save:
        filepath = os.path.join(output_dir, 'test_conditions.png')
        fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"   - {filepath}")
    
    # Résultats par zone
    for zone_name, results in all_results.items():
        fig = plot_zone_results(zone_name, results)
        figures[f'zone_{zone_name}'] = fig
        if save:
            filepath = os.path.join(output_dir, f'{zone_name}.png')
            fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
            print(f"   - {filepath}")
    
    # Comparaison globale
    fig = plot_comparison(all_results, zone_empilements)
    figures['comparison'] = fig
    if save:
        filepath = os.path.join(output_dir, 'comparison.png')
        fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"   - {filepath}")
    
    # -------------------------------------------------------------------------
    # NOUVELLES VISUALISATIONS: Classement des essais par matériau
    # -------------------------------------------------------------------------
    print("\n   Génération des matrices de classement...")
    
    # Matrices de classement par zone (température moyenne)
    ranking_figs, analysis, rankings = plot_all_ranking_matrices(
        all_results, n_top=n_top_ranking, metric='mean'
    )
    
    for zone_name, fig in ranking_figs.items():
        figures[f'ranking_{zone_name}'] = fig
        if save:
            filepath = os.path.join(output_dir, f'ranking_{zone_name}.png')
            fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
            print(f"   - {filepath}")
    
    # Résumé global des classements
    fig = plot_global_ranking_summary(analysis, rankings, n_top=n_top_ranking)
    figures['ranking_summary'] = fig
    if save:
        filepath = os.path.join(output_dir, 'ranking_summary.png')
        fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"   - {filepath}")
    
    if show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)
    
    return figures
