/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    object      k;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// NACA0012 at Re_c = 200,000, AoA = 5 degrees
// Turbulence intensity: I = 0.5%  (low freestream, matches LES reference conditions)
// U_inf = 1 m/s
// k = 1.5 * (I * U_inf)^2 = 1.5 * (0.005 * 1.0)^2 = 3.75e-5 m^2/s^2
//
// Wall BC: kqRWallFunction — zero-gradient-like, consistent with low-Re SST
// Patch names match blockMeshDict: inlet, outlet, aerofoil, back, front

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 3.75e-05;

boundaryField
{
    // ── Full farfield C-mesh boundary ─────────────────────────────────────────
    inlet
    {
        type            freestream;
        freestreamValue uniform 3.75e-05;
    }

    // ── Wake / trailing-edge exit boundary ───────────────────────────────────
    outlet
    {
        type            zeroGradient;
    }

    // ── Airfoil surface ───────────────────────────────────────────────────────
    // kqRWallFunction: applies wall function for k (sets k~0 at wall)
    // Appropriate for low-Re k-omega SST with y+~1
    aerofoil
    {
        type            kqRWallFunction;
        value           uniform 0;
    }

    // ── Spanwise empty patches (2-D simulation) ───────────────────────────────
    back
    {
        type            empty;
    }

    front
    {
        type            empty;
    }
}

// ************************************************************************* //
