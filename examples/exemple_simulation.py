"""
Exemple de simulation thermique multicouche
===========================================

Ce script montre comment utiliser le solveur et la bibliothèque de matériaux
pour simuler la conduction thermique transitoire dans un empilement multicouche.

Cas simulé:
- Empilement: Acier inoxydable 304 (5 mm) + Isolant céramique (20 mm) + Aluminium 6061 (10 mm)
- Condition gauche: Convection + Rayonnement (données du fichier flux.csv)
- Condition droite: Adiabatique
- Température initiale: 300 K

Nouveauté: Prise en compte du rayonnement avec l'environnement
    Flux radiatif = σ·ε·(T_surface⁴ - T_surroundings⁴)
    où σ = 5.67e-8 W/(m²·K⁴) est la constante de Stefan-Boltzmann
"""

import numpy as np
import matplotlib.pyplot as plt

# Import du solveur et de la bibliothèque de matériaux
from solver import ThermalSolver1D, compute_radiation_flux, compute_radiation_coefficient, STEFAN_BOLTZMANN
from material_library import get_materials, MaterialLibrary


def main():
    """Simulation principale."""
    
    print("="*70)
    print("SIMULATION THERMIQUE MULTICOUCHE")
    print("Avec convection et rayonnement")
    print("="*70)
    
    # ========================================================================
    # 1. DÉFINITION DES MATÉRIAUX
    # ========================================================================
    
    print("\n1. Chargement des matériaux depuis la bibliothèque...")
    
    # Charger les matériaux depuis la bibliothèque
    material_data = get_materials('steel_304', 'ceramic_fiber', 'aluminum_6061')
    
    # Afficher les propriétés à température ambiante
    print("\n   Propriétés à 300 K:")
    for name in ['steel_304', 'ceramic_fiber', 'aluminum_6061']:
        k, rho, cp = MaterialLibrary.get_properties(name, 300)
        alpha = k / (rho * cp)
        print(f"   - {name}: k={k:.2f} W/(m·K), ρ={rho:.0f} kg/m³, "
              f"cp={cp:.0f} J/(kg·K), α={alpha:.2e} m²/s")
    
    # ========================================================================
    # 2. DÉFINITION DE LA GÉOMÉTRIE
    # ========================================================================
    
    print("\n2. Définition de la géométrie...")
    
    layers = [
        {'material': 'steel_304', 'thickness': 0.005},      # 5 mm
        {'material': 'ceramic_fiber', 'thickness': 0.020},  # 20 mm
        {'material': 'aluminum_6061', 'thickness': 0.010}   # 10 mm
    ]
    
    L_total = sum(layer['thickness'] for layer in layers)
    
    print(f"\n   Configuration:")
    for i, layer in enumerate(layers):
        print(f"   - Couche {i+1}: {layer['material']}, {layer['thickness']*1000:.0f} mm")
    print(f"   - Épaisseur totale: {L_total*1000:.0f} mm")
    
    # ========================================================================
    # 3. CRÉATION DU SOLVEUR
    # ========================================================================
    
    print("\n3. Création du solveur...")
    
    Nx = 101  # Nombre de nœuds
    solver = ThermalSolver1D(layers, material_data, Nx=Nx)
    
    print(f"   - Nombre de nœuds: {Nx}")
    print(f"   - Pas spatial: {solver.dx*1000:.3f} mm")
    
    # ========================================================================
    # 4. LECTURE DES CONDITIONS LIMITES
    # ========================================================================
    
    print("\n4. Lecture des conditions limites (flux.csv)...")
    
    data = np.loadtxt('flux.csv', delimiter=',', skiprows=1)
    t_flux = data[:, 0]
    h_data = data[:, 1]
    T_inf_data = data[:, 2]
    T_s_data = data[:, 3]  # Température de l'environnement pour le rayonnement
    
    print(f"   - Temps: {t_flux[0]:.0f} s → {t_flux[-1]:.0f} s")
    print(f"   - h (convection): {h_data.min():.0f} → {h_data.max():.0f} W/(m²·K)")
    print(f"   - T_inf (convection): {T_inf_data.min():.0f} → {T_inf_data.max():.0f} K")
    print(f"   - T_s (rayonnement): {T_s_data.min():.0f} → {T_s_data.max():.0f} K")
    
    # Émissivité de la surface (acier inox poli)
    epsilon = 0.3
    print(f"   - Émissivité ε: {epsilon}")
    
    # Fonctions d'interpolation pour les CL
    def bc_left(t):
        """Condition limite gauche: convection + rayonnement."""
        h = np.interp(t, t_flux, h_data)
        T_inf = np.interp(t, t_flux, T_inf_data)
        T_s = np.interp(t, t_flux, T_s_data)
        return {
            'type': 'convection_radiation',
            'h': h,
            'T_inf': T_inf,
            'epsilon': epsilon,
            'T_s': T_s
        }
    
    def bc_right(t):
        """Condition limite droite: adiabatique."""
        return {'type': 'adiabatic'}
    
    # ========================================================================
    # 5. SIMULATION
    # ========================================================================
    
    print("\n5. Lancement de la simulation...")
    
    T_init = 300.0  # Température initiale [K]
    t_end = t_flux[-1]
    dt = 0.1  # Pas de temps [s]
    
    print(f"   - T_init: {T_init} K")
    print(f"   - Durée: {t_end} s")
    print(f"   - Pas de temps: {dt} s")
    print(f"   - Nombre de pas: {int(t_end/dt)}")
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, 
                          save_every=10, verbose=True)
    
    print(f"\n   Simulation terminée!")
    print(f"   - {len(result['time'])} instants sauvegardés")
    print(f"   - T_min = {result['T'].min():.2f} K")
    print(f"   - T_max = {result['T'].max():.2f} K")
    
    # ========================================================================
    # 6. CALCUL DES FLUX
    # ========================================================================
    
    print("\n6. Analyse des flux thermiques...")
    
    # Calculer les flux à la surface gauche pour chaque instant sauvegardé
    flux_conv = []
    flux_rad = []
    flux_total = []
    h_rad_values = []
    
    for i, t in enumerate(result['time']):
        T_surface = result['T'][i, 0]
        h = np.interp(t, t_flux, h_data)
        T_inf = np.interp(t, t_flux, T_inf_data)
        T_s = np.interp(t, t_flux, T_s_data)
        
        # Flux convectif (positif si chaleur entre dans le matériau)
        q_conv = h * (T_inf - T_surface)
        
        # Flux radiatif (positif si chaleur entre dans le matériau)
        q_rad = STEFAN_BOLTZMANN * epsilon * (T_s**4 - T_surface**4)
        
        # Coefficient radiatif équivalent
        h_rad = compute_radiation_coefficient(T_surface, epsilon)
        
        flux_conv.append(q_conv)
        flux_rad.append(q_rad)
        flux_total.append(q_conv + q_rad)
        h_rad_values.append(h_rad)
    
    flux_conv = np.array(flux_conv)
    flux_rad = np.array(flux_rad)
    flux_total = np.array(flux_total)
    h_rad_values = np.array(h_rad_values)
    
    print(f"   - Flux convectif final: {flux_conv[-1]:.1f} W/m²")
    print(f"   - Flux radiatif final: {flux_rad[-1]:.1f} W/m²")
    print(f"   - Flux total final: {flux_total[-1]:.1f} W/m²")
    print(f"   - h_rad à T={result['T'][-1, 0]:.0f}K: {h_rad_values[-1]:.1f} W/(m²·K)")
    
    # ========================================================================
    # 7. VISUALISATION
    # ========================================================================
    
    print("\n7. Génération des graphiques...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # --- Graphique 1: Profils de température ---
    ax = axes[0, 0]
    
    n_profiles = 6
    indices = np.linspace(0, len(result['time'])-1, n_profiles, dtype=int)
    colors = plt.cm.hot(np.linspace(0.2, 0.8, n_profiles))
    
    for idx, color in zip(indices, colors):
        t = result['time'][idx]
        T = result['T'][idx, :]
        ax.plot(solver.x*1000, T, '-', color=color, linewidth=2, label=f't={t:.0f}s')
    
    # Marquer les interfaces
    cum_thick = np.cumsum([layer['thickness'] for layer in layers])
    for x_int in cum_thick[:-1]:
        ax.axvline(x=x_int*1000, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Profils de température')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Graphique 2: Évolution temporelle ---
    ax = axes[0, 1]
    
    x_positions = [0, 0.005, 0.015, 0.025, 0.035]
    labels = ['Surface gauche', 'Interface 1', 'Milieu isolant', 'Interface 2', 'Surface droite']
    
    for x_pos, label in zip(x_positions, labels):
        idx = np.argmin(np.abs(solver.x - x_pos))
        ax.plot(result['time'], result['T'][:, idx], '-', linewidth=2, label=label)
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Évolution temporelle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Graphique 3: Conditions limites ---
    ax = axes[0, 2]
    ax2 = ax.twinx()
    
    l1 = ax.plot(t_flux, h_data, 'b-', linewidth=2, marker='o', label='h (convection)')
    l2 = ax2.plot(t_flux, T_inf_data, 'r-', linewidth=2, marker='s', label='T_inf (convection)')
    l3 = ax2.plot(t_flux, T_s_data, 'g--', linewidth=2, marker='^', label='T_s (rayonnement)')
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('h [W/(m²·K)]', color='b')
    ax2.set_ylabel('Température [K]', color='r')
    ax.set_title('Conditions limites')
    
    lines = l1 + l2 + l3
    labels_leg = [l.get_label() for l in lines]
    ax.legend(lines, labels_leg, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Graphique 4: Flux thermiques ---
    ax = axes[1, 0]
    
    ax.plot(result['time'], flux_conv, 'b-', linewidth=2, label='Convection')
    ax.plot(result['time'], flux_rad, 'r-', linewidth=2, label='Rayonnement')
    ax.plot(result['time'], flux_total, 'k-', linewidth=2.5, label='Total')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Flux [W/m²]')
    ax.set_title('Flux thermiques à la surface gauche')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Graphique 5: Coefficients de transfert ---
    ax = axes[1, 1]
    
    h_interp = np.interp(result['time'], t_flux, h_data)
    
    ax.plot(result['time'], h_interp, 'b-', linewidth=2, label='h convection')
    ax.plot(result['time'], h_rad_values, 'r-', linewidth=2, label='h radiatif (linéarisé)')
    ax.plot(result['time'], h_interp + h_rad_values, 'k--', linewidth=2, label='h total')
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Coefficient [W/(m²·K)]')
    ax.set_title('Coefficients de transfert')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Graphique 6: Carte de température ---
    ax = axes[1, 2]
    
    T_grid = result['T']
    extent = [solver.x[0]*1000, solver.x[-1]*1000, result['time'][0], result['time'][-1]]
    
    im = ax.imshow(T_grid, aspect='auto', origin='lower', extent=extent,
                   cmap='hot', interpolation='bilinear')
    
    for x_int in cum_thick[:-1]:
        ax.axvline(x=x_int*1000, color='white', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Temps [s]')
    ax.set_title('Carte spatio-temporelle')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('T [K]')
    
    plt.suptitle('Simulation thermique multicouche avec convection + rayonnement',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('simulation_multicouche.png', dpi=150)
    plt.show()
    
    print(f"\n   Figure sauvegardée: simulation_multicouche.png")
    
    # ========================================================================
    # 8. RÉSUMÉ
    # ========================================================================
    
    print("\n" + "="*70)
    print("RÉSUMÉ DE LA SIMULATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Empilement: {' + '.join([l['material'] for l in layers])}")
    print(f"  - Épaisseur totale: {L_total*1000:.0f} mm")
    print(f"  - Maillage: {Nx} nœuds, dx = {solver.dx*1000:.3f} mm")
    print(f"\nConditions limites:")
    print(f"  - Gauche: Convection (h variable) + Rayonnement (ε={epsilon})")
    print(f"  - Droite: Adiabatique")
    print(f"\nSimulation:")
    print(f"  - Durée: {t_end} s")
    print(f"  - Pas de temps: {dt} s")
    print(f"\nRésultats:")
    print(f"  - T initiale: {T_init} K")
    print(f"  - T finale min: {result['T'][-1].min():.2f} K")
    print(f"  - T finale max: {result['T'][-1].max():.2f} K")
    print(f"  - T finale moyenne: {result['T'][-1].mean():.2f} K")
    print(f"\nFlux à la surface (final):")
    print(f"  - Convection: {flux_conv[-1]:.1f} W/m²")
    print(f"  - Rayonnement: {flux_rad[-1]:.1f} W/m²")
    print(f"  - Total: {flux_total[-1]:.1f} W/m²")
    print("="*70)


if __name__ == '__main__':
    main()
