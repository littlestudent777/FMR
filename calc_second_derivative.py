import sympy as sym


def main():
    # Define symbols
    x, y, z = sym.symbols('x y z')
    theta, phi = sym.symbols('theta phi', positive=True)
    H1, H2, H3 = sym.symbols('H[0] H[1] H[2]')
    N1, N2, N3 = sym.symbols('N[0] N[1] N[2]')
    K1, K2 = sym.symbols('K1 K2')
    Ms = sym.symbols('Ms')

    # Coordinate transformation
    cartesian_to_spherical = {
        x: sym.sin(theta) * sym.cos(phi),
        y: sym.sin(theta) * sym.sin(phi),
        z: sym.cos(theta)
    }

    # Define functions
    functions = {
        "F_ext": -Ms * (x * H1 + y * H2 + z * H3),
        "F_ani": K1 * (x ** 2 * y ** 2 + y ** 2 * z ** 2 + z ** 2 * x ** 2) + K2 * x ** 2 * y ** 2 * z ** 2,
        "F_demag": 2 * sym.pi * Ms ** 2 * (x ** 2 * N1 + y ** 2 * N2 + z ** 2 * N3)
    }

    # Process functions
    results = {}
    for name, fun in functions.items():
        # Convert to spherical coordinates
        fun_sph = fun.subs(cartesian_to_spherical).simplify()

        # Calculate derivatives
        d2_dtheta2 = sym.diff(fun_sph, theta, 2).simplify()
        d2_dphi2 = sym.diff(fun_sph, phi, 2).simplify()
        d2_dthetadphi = sym.diff(fun_sph, theta, phi).simplify()

        # Verify symmetry of mixed derivatives
        symmetry_check = sym.simplify(d2_dthetadphi - sym.diff(fun_sph, phi, theta)) == 0

        # Store results
        results[name] = {
            'spherical': fun_sph,
            'd2_dtheta2': d2_dtheta2,
            'd2_dphi2': d2_dphi2,
            'd2_dthetadphi': d2_dthetadphi,
            'symmetry_check': symmetry_check
        }

    # Print results in a structured way
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print("Spherical form:", res['spherical'])
        print("d²/dθ²:", res['d2_dtheta2'])
        print("d²/dφ²:", res['d2_dphi2'])
        print("d²/dθdφ:", res['d2_dthetadphi'])
        print("Symmetry check:", res['symmetry_check'])


if __name__ == "__main__":
    main()
