# Credits for the SOMA and MyoSuite videos

Anatomy and learned deformation: **SOMA and SKIM v2, subject S1**, by Eduardo
Alvarado and the SOMA authors. The downloaded release includes the MIT license
reproduced in `SOMA_ASSETS_LICENSE.txt`.

- [SOMA project](https://vcai.mpi-inf.mpg.de/projects/SOMA/)
- [SOMA source and release documentation](https://github.com/edualvarado/SOMA)

Research citation: Eduardo Alvarado, Emily Kim, Gerrit Nolte, Friedemann Runte,
Mario Botsch, Marc Habermann and Christian Theobalt. *SOMA: From Surface
Observations to Muscle Anatomy.* European Conference on Computer Vision, 2026.

Musculoskeletal simulation and published elbow/walking policies: **MyoSuite**
and **MyoSim**, with **MuJoCo**. The policy source is the MyoSuite 3 baseline
release, revision `v3.0`.

- [MyoSuite](https://github.com/MyoHub/myosuite)
- [MyoSuite baseline policies](https://huggingface.co/myohub/myosuite-3-baselines)
- [MuJoCo](https://mujoco.org/)

The full-body diagnostic uses **musclemimic_models 1.0.6**:
[model project](https://github.com/amathislab/musclemimic_models).
Its motion is a generated mobility sweep. The walking video uses MyoSuite's
bilateral leg model and actual policy control.

Rendering: **Blender 4.5.3 LTS**, Eevee; typography, charts and video composition
generated from the recorded states. The videos contain no music.

Suggested short attribution when sharing:

> SOMA / SKIM S1 anatomy integrated with MyoSuite and MuJoCo. Pose-conditioned
> anatomical visualization with approximate registration; activation colors
> come from simulated muscle states. SOMA: Alvarado et al., ECCV 2026.

The two new registration comparisons use recorded SKIM S1 shot 006, frames 650–725, retargeted to the same full-body model by bounded kinematic IK. They use uniform anatomy colors and rest activation. Reference grid, labels and height plots show raw and final version 3 anatomy on identical native kinematics. Recorded motion is not a contact-stable control policy.
