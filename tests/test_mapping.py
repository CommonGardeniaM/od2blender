from mmd_trace.config import default_bone_map
from mmd_trace.retarget.mapping import BoneMapping


def test_vector_pairs():
    mapping = BoneMapping(default_bone_map())
    pairs = mapping.vector_pairs()
    assert ("upper_body", "pelvis", "spine") in pairs
