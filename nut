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
    object      nut;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// NACA0012 at Re_c = 200,000, AoA = 5 degrees
// nut = k / omega = 3.75e-5 / 75 = 5e-7 m^2/s
// nut/nu = 5e-7 / 5e-6 = 0.1  (very clean freestream)
//
// At wall: nut is calculated by nutkWallFunction from k and y+
// At farfield/outlet: 'calculated' — derived from k and omega fields
//
// Patch names match blockMeshDict: inlet, outlet, aerofoil, back, front

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 5e-07;

boundaryField
{
    // ── Full farfield C-mesh boundary ─────────────────────────────────────────
    inlet
    {
        type            calculated;
        value           uniform 5e-07;
    }

    // ── Wake / trailing-edge exit boundary ───────────────────────────────────
    outlet
    {
        type            calculated;
        value           uniform 5e-07;
    }

    // ── Airfoil surface ───────────────────────────────────────────────────────
    // nutkWallFunction: computes nut at wall from k and wall distance
    // Consistent with k-omega SST model at y+~1
    aerofoil
    {
        type            nutkWallFunction;
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
