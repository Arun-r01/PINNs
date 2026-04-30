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
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// NACA0012 at Re_c = 200,000, AoA = 5 degrees
// Kinematic pressure p/rho  [m^2/s^2]
// Gauge pressure: p_inf = 0 at freestream
//
// Patch names match blockMeshDict: inlet, outlet, aerofoil, back, front

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    // ── Full farfield C-mesh boundary ─────────────────────────────────────────
    inlet
    {
        type            freestreamPressure;
        freestreamValue uniform 0;   // p_inf = 0 (gauge)
    }

    // ── Wake / trailing-edge exit boundary ───────────────────────────────────
    outlet
    {
        type            fixedValue;
        value           uniform 0;   // reference pressure at exit
    }

    // ── Airfoil surface ───────────────────────────────────────────────────────
    // zeroGradient: dp/dn = 0 (no normal pressure flux through wall)
    aerofoil
    {
        type            zeroGradient;
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
