"""MuJoCo enum decisions must survive integer scalars with strict equality."""

import mujoco
import numpy as np
import pytest

from myosuite_soma.soma import _source_poses
from myosuite_soma.soma_retarget import JointCouplings


class OpaqueInteger:
    def __init__(self, value):
        self.value = int(value)

    def __int__(self):
        return self.value

    def __eq__(self, other):
        return isinstance(other, OpaqueInteger) and self.value == other.value


class CompiledModelProxy:
    def __init__(self, model):
        self.compiled = model
        self.jnt_type = tuple(OpaqueInteger(value) for value in model.jnt_type)
        self.eq_type = tuple(OpaqueInteger(value) for value in model.eq_type)

    def __getattr__(self, name):
        return getattr(self.compiled, name)


@pytest.mark.parametrize("quaternion", [[0.0, 0, 0, 0], [2.0, 0, 0, 0]])
def test_free_quaternion_validation_survives_strict_scalar_equality(quaternion):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint name="root"/>'
        '<geom size=".1"/></body></worldbody></mujoco>'
    )
    proxy = CompiledModelProxy(model)
    assert proxy.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
    assert int(proxy.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    poses = np.tile(model.qpos0, (2, 1))
    np.testing.assert_array_equal(_source_poses(proxy, poses, "qpos"), poses)
    poses[1, 3:7] = quaternion
    with pytest.raises(ValueError, match="unit quaternion"):
        _source_poses(proxy, poses, "qpos")


def test_coupled_scalar_discovery_and_limits_survive_strict_scalar_equality():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
      <worldbody><body><joint name="parent" type="hinge" range="-1 1"/><geom size=".1"/>
        <body><joint name="child" type="slide" range="-.2 .2"/><geom size=".1"/></body>
      </body></worldbody><equality>
        <joint joint1="child" joint2="parent" polycoef="0 2 0 0 0"/>
      </equality></mujoco>""")
    proxy = CompiledModelProxy(model)
    assert proxy.eq_type[0] != mujoco.mjtEq.mjEQ_JOINT
    before = model.jnt_range.copy()
    domain = JointCouplings(proxy)
    assert domain.joints == [model.joint("parent").id]
    assert domain.order == [0]
    np.testing.assert_allclose(domain.bounds[:, 0], [-0.1, 0.1])
    pose = domain.apply(np.array([0.08, 0.0]))
    np.testing.assert_allclose(pose, [0.08, 0.16])
    np.testing.assert_array_equal(model.jnt_range, before)
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    equality = np.asarray(data.efc_type, dtype=int) == int(
        mujoco.mjtConstraint.mjCNSTR_EQUALITY
    )
    assert equality.any()
    np.testing.assert_allclose(data.efc_pos[equality], 0, atol=1e-14)
