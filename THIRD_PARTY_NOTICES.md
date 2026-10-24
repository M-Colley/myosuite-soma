# Third-party notices

## MyoSuite

The backend modules and the optional renderer adapter were developed from the
SOMA integration in a MyoSuite `dev` checkout based on commit
`c0af9c656de2aa1807924e63d0d5320d77aa1ebd`.

Copyright (c) MyoSuite Authors. All rights reserved.

MyoSuite source is licensed under the Apache License, Version 2.0. Its complete
license text is retained in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
This repository extracts the geometry backend into `myosuite_soma`, makes
MyoSuite optional, and supplies explicit adapter installation and independent
packaging. Original source notices are retained. This is an independent
integration and does not imply endorsement by the MyoSuite authors.

Upstream: [MyoHub/myosuite](https://github.com/MyoHub/myosuite).

## SOMA

The released predictor architecture, rigid corrective equations and skinning
adaptations follow the SOMA implementation at commit
`59c5a52c9e4dcaa5fbe4fdc4744c0c8781c64229`.

Copyright (c) 2026 Eduardo Alvarado and the SOMA authors.

The corresponding MIT License is reproduced unchanged in
[licenses/SOMA-MIT.txt](licenses/SOMA-MIT.txt). This notice covers the source
adaptations; it does not relicense separately distributed datasets, recordings,
geometry, trained weights or subjects' likenesses. Obtain such assets from the
authors and retain their accompanying terms. This repository does not include
raw assets or perform an authenticated asset download. Rendered demonstration
media is credited under `assets/`, with its supplied asset license retained.

Upstream: [edualvarado/SOMA](https://github.com/edualvarado/SOMA).
Project: [SOMA](https://vcai.mpi-inf.mpg.de/projects/SOMA/).

## Runtime dependencies and generated media

NumPy, SciPy, MuJoCo, OSQP, Torch, MyoSuite, USD and Blender are separate projects
with their own licenses and distributions. They are not incorporated here by
these notices. Optional examples refer to them by name. No third-party anatomy
mesh, simulation model, policy, release checkpoint or rendered subject video is
bundled in this source repository. The credited rendered contact sheet and
prerelease videos are derived demonstrations, separate from those raw assets.
