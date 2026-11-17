# COMSOL physics export: worked example

`Stage1TwoPort.java` is `benchmarks/designs/stage1_twoport.json` exported with
`axmp export-comsol ... --format java --physics fullwave`. axionMultiPhysics's numbers are in
`axionmultiphysics_reference.json`, and docs/comsol_export.md section 4 explains them.

On a machine with COMSOL (RF Module):

```bash
comsolcompile Stage1TwoPort.java
comsol batch -inputfile Stage1TwoPort.class   # -> Stage1TwoPort_impedance.txt, Stage1TwoPort_voltage.txt, .mph
python make_example.py OUT --compare .        # compare with axionMultiPhysics (open-circuit conversion included)
```

`python make_example.py OUT` regenerates both routes, including the mesh-import route
`stage1_twoport.mphtxt` + `Import_stage1_twoport.java`. That route is not committed (5 MB), and its Java
holds an absolute path to the .mphtxt. `--solve` re-solves with axionMultiPhysics (minutes; use a batch job on a
cluster).

Expected values, order-2 axionMultiPhysics (COMSOL convention, port 2 open):

| f [MHz] | Zport_1 [Ω] | \|V_open\| [V] | \|Vport_1\| at 50 Ω [V] |
|---|---|---|---|
| 1 | +3.2097i | 1.2518e-3 | 1.2492e-3 |
| 12 | +39.487i | 0.09560 | 0.07502 |

The first-order axionMultiPhysics values are 14 % lower in L, so axionMultiPhysics is not mesh-converged here. Expect
agreement at the ~10 % level only.
