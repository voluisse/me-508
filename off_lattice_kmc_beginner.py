import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import KDTree
import matplotlib.patches as mpatches

# ------- Parameters -------
N_atoms_initial = 2500      # initial number of atoms (approx 50x50)
Lx, Ly = 20.0, 20.0        # simulation box size in Angstroms (arbitrary units)
kT = 0.05                  # thermal energy in eV
p_formation = 0.5          # Probability of vacancy formation/annihilation event per step

# Vacancy formation energy scale (eV)
formation_energy_min = 0.025
formation_energy_max = 0.1

# Vacancy diffusion barrier energy scale (eV)
diffusion_energy_min = 0.01
diffusion_energy_max = 0.05

# Distance cutoff for neighbors (Angstrom)
neighbor_cutoff = 1.5

# ---- Generate initial 2D FCC (111)-like lattice coordinates (approximate triangular lattice) ----
def create_fcc111_lattice(n_atoms, Lx, Ly):
    # Approximate 2D triangular lattice for FCC(111) projection
    # Interatomic spacing ~1.0 (set unit length)
    spacing = 1.0
    nx = int(np.sqrt(n_atoms * Ly / Lx)) + 1
    ny = int(n_atoms / nx) + 1

    positions = []
    for i in range(nx):
        for j in range(ny):
            x = i * spacing + (j % 2) * spacing / 2
            y = j * spacing * np.sqrt(3)/2
            if x < Lx and y < Ly:
                positions.append([x, y])
    return np.array(positions)

positions = create_fcc111_lattice(N_atoms_initial, Lx, Ly)
N_atoms = len(positions)
print(f"Initial atoms: {N_atoms}")

# Initially no vacancies: vacancies = []
vacancies = np.empty((0,2))   # no vacancy positions yet

# Assign random vacancy formation energies per initial atom
formation_energies = np.random.uniform(formation_energy_min, formation_energy_max, N_atoms)

# Assign random diffusion activation energies per atom (for diffusion events)
diffusion_energies = np.random.uniform(diffusion_energy_min, diffusion_energy_max, N_atoms)

# Use KDTree for neighbor search on atomic positions
tree = KDTree(positions)

# ----- Data to track -----
vacancy_counts = []
diffusion_event_counts = []
cumulative_diffusions = 0

# Snapshots for animation: each snapshot stores atom positions and vacancy positions
atom_snapshots = []
vacancy_snapshots = []

# Event highlight masks for animation (arrays storing indices of atoms/vacancies involved in events)
atom_colors = []  # rgb colors for atoms in each frame
vacancy_colors = []  # rgb colors for vacancies in each frame

# ------- Helper functions -------

def metropolis_acceptation(delta_E):
    """Metropolis criterion acceptance"""
    # If the energy change delta_E is negative or zero, accept move unconditionally (energy decreases or unchanged)
    if delta_E <= 0:
        return True
    else:
        # If energy increases, accept move with probability exp(-delta_E/kT) to satisfy detailed balance
        return np.random.rand() < np.exp(-delta_E / kT)

# ------- Vacancy formation -------
def attempt_vacancy_formation():
    """Attempt to create a vacancy by removing a random atom"""
    global positions, formation_energies, diffusion_energies, vacancies

    # If no atoms exist, cannot form vacancy, reject immediately
    if len(positions) == 0:
        return False, [], []

    # Randomly select one atom index from current atoms
    idx = np.random.randint(len(positions))

    # Get vacancy formation energy associated with this atom (energy cost to remove it)
    delta_E = formation_energies[idx]  # The energy cost for removing the atom at index idx

    # Decide whether to accept the move with Metropolis acceptance criterion
    accepted = metropolis_acceptation(delta_E)

    # Prepare empty lists to record formed atom and vacancy indices (for visualization/event tracking)
    formed_sites_atom_idx = []    # vacancy formation removes an atom, so no new atom
    formed_sites_vac_idx = []     # will hold index of newly created vacancy if accepted

    if accepted:
        # The atom at idx is removed, creating a vacancy at that position
        vac_pos = positions[idx]

        # Append vacancy coordinate to vacancies array (stack vertically)
        vacancies = np.vstack([vacancies, vac_pos])

        # Remove atom from positions array by deleting its entry
        positions_new = np.delete(positions, idx, axis=0)
        formation_energies_new = np.delete(formation_energies, idx)
        diffusion_energies_new = np.delete(diffusion_energies, idx)

        # Update system arrays with atom removed
        positions = positions_new
        formation_energies = formation_energies_new
        diffusion_energies = diffusion_energies_new

        # Record index of new vacancy for tracking: it's the last vacancy appended
        formed_sites_vac_idx.append(len(vacancies) - 1)

    # Return whether move accepted, empty atom list since no new atom formed, and vacancy indices formed
    return accepted, [], formed_sites_vac_idx

# ------- Vacancy annihilation -------
def attempt_vacancy_annihilation():
    """Attempt to fill a vacancy by adding an atom back"""
    global vacancies, positions, formation_energies, diffusion_energies

    # If no vacancies exist, cannot fill a vacancy, reject immediately
    if len(vacancies) == 0:
        return False, [], []

    # Randomly select vacancy index to fill
    idx = np.random.randint(len(vacancies))

    # Get vacancy position coordinate to add atom there
    vac_pos = vacancies[idx]

    # Differs from formation: energy gain from removing vacancy (assumed negative formation energy)
    delta_E = -np.mean(formation_energies)  # The energy gain from adding an atom back to a vacancy

    # Determine acceptance probability by Metropolis criterion
    accepted = metropolis_acceptation(delta_E)

    # Lists for tracking newly formed atoms and vacancies for visualization/events
    formed_sites_atom_idx = []
    formed_sites_vac_idx = []

    if accepted:
        # Add atom coordinate to positions array at vacancy location
        positions = np.vstack([positions, vac_pos])

        # Generate new random energy parameters for this added atom
        formation_energies = np.append(formation_energies,
                                       np.random.uniform(formation_energy_min, formation_energy_max))

        diffusion_energies = np.append(diffusion_energies,
                                       np.random.uniform(diffusion_energy_min, diffusion_energy_max))

        # Remove vacancy from vacancies array
        vacancies = np.delete(vacancies, idx, axis=0)

        # Track the newly formed atom index as the last atom appended
        formed_sites_atom_idx.append(len(positions) - 1)

    # Return move acceptance, list of new atoms added, and empty vacancy list (vacancy removed)
    return accepted, formed_sites_atom_idx, []

# ------- Vacancy diffusion -------
def attempt_vacancy_diffusion():
    """Attempt vacancy diffusion:
    - Pick a random vacancy,
    - Find all atoms within neighbor cutoff,
    - Choose random atom neighbor,
    - Proposal: atom moves into vacancy (vacancy moves to atom's old pos)
    - Metropolis acceptance based on diffusion barrier associated with atom."""
    
    global positions, vacancies, diffusion_energies, cumulative_diffusions

    # Cannot diffuse if system has no vacancies or no atoms
    if len(vacancies) == 0 or len(positions) == 0:
        return False, [], []

    # Randomly select a vacancy index
    v_idx = np.random.randint(len(vacancies))

    # Vacancy coordinate selected
    vac_pos = vacancies[v_idx]

    # Build spatial KDTree for atom positions to quickly find neighbors
    tree = KDTree(positions)

    # Query all atom indices within neighbor_cutoff distance of vacancy position
    nearby_atom_indices = tree.query_ball_point(vac_pos, neighbor_cutoff)

    # If no atoms near vacancy, diffusion hop impossible, reject
    if len(nearby_atom_indices) == 0:
        return False, [], []

    # Choose one random neighbor atom index among atoms near vacancy
    a_idx = np.random.choice(nearby_atom_indices)

    # Position of selected atom
    atom_pos = positions[a_idx] 

    # Diffusion barrier energy associated with chosen atom (activation energy for hop)
    delta_E = formation_energies[a_idx]  # The energy cost for the atom being moved into the vacancy

    # Apply Metropolis acceptance for diffusion attempt based on barrier
    accepted = metropolis_acceptation(delta_E)

    if accepted:
        # Swap positions: atom moves into vacancy position
        # Vacancy takes atom’s old coordinate
        vacancies[v_idx] = atom_pos
        positions[a_idx] = vac_pos

        # Increase cumulative diffusion event counter for bookkeeping
        cumulative_diffusions += 1

        # Return that diffusion accepted, with involved atom and vacancy indices for tracking/highlighting
        return True, [a_idx], [v_idx]
    else:
        # Move rejected: no changes; empty event lists
        return False, [], []

# ------- KMC step -------
def kmc_step():
    """Perform a single KMC step: Choose event type with probability p_formation"""
    global vacancies

    if np.random.rand() < p_formation:
        # Try formation or annihilation
        # Remove atom with 50% prob, else fill vacancy if any exist
        if len(vacancies) == 0:
            return attempt_vacancy_formation()
        if len(positions) == 0:
            return attempt_vacancy_annihilation()
        if np.random.rand() < 0.5:
            return attempt_vacancy_formation()
        else:
            return attempt_vacancy_annihilation()
    else:
        return attempt_vacancy_diffusion()

# ------- Simulation loop -------

N_frames = 5000
N_steps = 5000
steps_per_frame = N_steps // N_frames

# For animation event coloring:
atom_event_colors = []
vac_event_colors = []

print("Starting off-lattice vacancy KMC simulation ...")

for frame in range(N_frames):
    atoms_involved_positions = []
    vacs_involved_positions = []

    for _ in range(steps_per_frame):
        accepted, atoms_idx, vacs_idx = kmc_step()
        if accepted:
            for aidx in atoms_idx:
                if aidx < len(positions):
                    atoms_involved_positions.append(positions[aidx].copy())
            for vidx in vacs_idx:
                if vidx < len(vacancies):
                    vacs_involved_positions.append(vacancies[vidx].copy())

    atom_colors_frame = np.ones((len(positions), 3))  # white default
    vac_colors_frame = np.zeros((len(vacancies), 3))  # black default

    for pos in atoms_involved_positions:
        diffs = np.linalg.norm(positions - pos, axis=1)
        matches = np.where(diffs < 1e-8)[0]
        for m in matches:
            atom_colors_frame[m] = [0, 0, 1]  # blue highlight

    for pos in vacs_involved_positions:
        diffs = np.linalg.norm(vacancies - pos, axis=1)
        matches = np.where(diffs < 1e-8)[0]
        for m in matches:
            vac_colors_frame[m] = [1, 0, 0]  # red highlight

    atom_snapshots.append(positions.copy())
    vacancy_snapshots.append(vacancies.copy())
    atom_event_colors.append(atom_colors_frame)
    vac_event_colors.append(vac_colors_frame)

    vacancy_counts.append(len(vacancies))
    diffusion_event_counts.append(cumulative_diffusions)

    if frame % 10 == 0 or frame == N_frames - 1:
        print(f"Frame {frame+1}/{N_frames}: Vacancies = {len(vacancies)}, Diffusions = {cumulative_diffusions}")

# ------- Animation -------

fig, ax = plt.subplots(figsize=(6, 6))

def animate(frame):
    ax.clear()

    atoms = atom_snapshots[frame]
    vacs = vacancy_snapshots[frame]

    # Create dummy artists:
    legend_handles = [
        mpatches.Patch(color='white', label='Atoms', edgecolor='k'),
        mpatches.Patch(facecolor='yellow', edgecolor='grey', label='Vacancy (grey circle)'),
        mpatches.Patch(color='blue', label='Vacancy formation'),
        mpatches.Patch(color='green', label='Vacancy annihilation'),
        mpatches.Patch(color='red', label='Vacancy diffusion')
    ]

    ax.legend(handles=legend_handles, loc='upper right')

    # Plot atoms first
    ax.scatter(atoms[:, 0], atoms[:, 1], c=atom_event_colors[frame], s=20, edgecolors='k', label='Atoms')

    # Plot all vacancies (grey circles) ON TOP of atoms
    if len(vacs) > 0:
        ax.scatter(vacs[:, 0], vacs[:, 1],
                   facecolors='yellow',
                   edgecolors='grey',
                   s=150,
                   linewidths=2,
                   alpha=1.0,
                   label='Vacancies')

    # Plot red squares for diffusion-highlighted vacancies ON TOP of grey circles
    vacancy_colors = vac_event_colors[frame]
    red_indices = np.where(np.all(vacancy_colors == [1.0, 0.0, 0.0], axis=1))[0]
    if len(red_indices) > 0:
        ax.scatter(vacs[red_indices, 0], vacs[red_indices, 1], c='red', s=80, marker='s', label='Diffusing Vacancies')

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_title(f"KMC Step ~{(frame+1)*steps_per_frame}\nVacancies: {vacancy_counts[frame]}, Diffusions: {diffusion_event_counts[frame]}")
    ax.grid(True)

anim = animation.FuncAnimation(fig, animate, frames=N_frames, interval=200)
# anim.save("off_lattice_vacancy_kmc.mp4", writer='ffmpeg', fps=5)
# plt.close()

# ------- Plot vacancy and diffusion counts -------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

steps_array = np.arange(len(vacancy_counts)) * steps_per_frame

ax1.plot(steps_array, vacancy_counts, '-o', color='blue')
ax1.set_xlabel('KMC Steps')
ax1.set_ylabel('Number of Vacancies')
ax1.set_title('Vacancy Count vs KMC Step')
ax1.grid(True)

ax2.plot(steps_array, diffusion_event_counts, '-o', color='red')
ax2.set_xlabel('KMC Steps')
ax2.set_ylabel('Cumulative Diffusion Events')
ax2.set_title('Diffusion Events vs KMC Step')
ax2.grid(True)

plt.tight_layout()
plt.show()
