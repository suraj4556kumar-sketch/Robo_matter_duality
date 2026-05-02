Nbots: int = 100
    Nmag: int = 6

    # Geometry
    R_bot: float = 1.0
    R_ring: float = 0.78
    r_mag: float = 0.105

    # Larger box -> more open space for movement
    box: float = 20.0

    # Translational motion
    # Increased compared with previous code
    v_base: float = 8.00
    v_light_gain: float = 0.55

    # Light-gradient drift strength
    # This makes bots move toward the bright spot, not only rotate faster there.
    light_drift_strength: float = 1.20

    # Chiral rotation
    omega_base: float = 7.00
    omega_light_gain: float = 2.20

    # Noise
    Dt: float = 0.012
    Dr: float = 0.025
    Dphi: float = 0.012

    # Mobilities
    mu_t: float = 1.0
    mu_r: float = 0.50
    mu_phi: float = 1.0

    # Magnetic interaction
    k_m: float = 1.15
    r0: float = 1.40
    r_cut: float = 1.45

    # Steric repulsion
    k_rep: float = 200.0

    # Time
    dt: float = 0.015
    steps: int = 1600
    seed: int = 7