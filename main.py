import calc_fmr as my

experiments = [
    {
        "description": "External field only (H = [600, 250, 1000] G)",
        "m": [0.50306325, 0.20961179, 0.83844515],
        "fields": [my.ExternalField(H=[600.0, 250.0, 1000.0])]
    },
    {
        "description": "External field only (H = [800, 1000, 350] G)",
        "m": [0.60259486, 0.75324358, 0.26363525],
        "fields": [my.ExternalField(H=[800.0, 1000.0, 350.0])]
    },
    {
        "description": "Anisotropy",
        "m": [1.0, 0.0, 0.0],
        "fields": [my.Anisotropy()]
    },
    {
        "description": "Anisotropy (K1 = -4.2e5, K2 = 1.0e5)",
        "m": [0.57735, 0.57735, 0.57735],
        "fields": [my.Anisotropy(K1_val=-4.2e5, K2_val=1.0e5)]
    },
    {
        "description": "Demagnetization field (Cylinder, z is inf)",
        "m": [0.0, 0.0, 1.0],
        "fields": [my.DemagnetizingField(N=[0.5, 0.5, 0.0])]
    },
    {
        "description": "Combination of anisotropy, external field, and demagnetizing field",
        "m": [0.97852257, 0.19680969, 0.0613149],
        "fields": [
            my.Anisotropy(),
            my.ExternalField(H=[800, 250, 1000]),
            my.DemagnetizingField(N=[0.1, 0.1, 0.8])
        ]
    }
]

# Run experiments
for exp in experiments:
    print(f"\n{exp['description']}")
    print(f"Magnetization direction: {exp['m']}")

    try:
        omega = my.calculate_fmr_frequency(
            m=exp['m'],
            fields=exp['fields']
        )
        print(f"FMR frequency: {omega:.3f} Ghz")
    except ValueError as e:
        print(f"Error: {e}")
