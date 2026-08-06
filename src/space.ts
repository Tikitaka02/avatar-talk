import * as THREE from "three";
import type { VRM, VRMHumanBoneName } from "@pixiv/three-vrm";
import type { Landmark } from "@mediapipe/tasks-vision";

/**
 * How far to trust MediaPipe's depth. A single camera cannot measure z, so the
 * model infers it — and it drifts badly: a symmetric pose can report one elbow
 * twice as far forward as the other. Shrinking z pulls limbs back toward the
 * silhouette the camera actually saw, which is the part it gets right.
 */
export const Z_TRUST = 0.45;

/**
 * MediaPipe world landmarks are metric, with x to the image right, y down and
 * z growing away from the camera. VRM 1.0 is y up and +z forward, so y and z
 * flip. x flips too, which mirrors the pose: raise your right hand and the
 * avatar raises the hand on the same side of the screen, like a mirror. That
 * mirroring is also why right-side landmarks drive the avatar's left bones.
 */
export function toAvatarSpace(
  lm: Landmark,
  out: THREE.Vector3,
  zTrust = Z_TRUST
): THREE.Vector3 {
  return out.set(-lm.x, -lm.y, -lm.z * zTrust);
}

/**
 * World rotation of a joint, from the local rotations above it. In the
 * normalized humanoid rig every bone rests axis-aligned with the world, so
 * multiplying the local rotations along a chain gives that joint's world
 * rotation without touching matrices.
 */
export function chainWorldQuaternion(
  vrm: VRM,
  chain: VRMHumanBoneName[],
  out: THREE.Quaternion
): THREE.Quaternion {
  out.identity();
  for (const name of chain) {
    const node = vrm.humanoid.getNormalizedBoneNode(name);
    if (node) out.multiply(node.quaternion);
  }
  return out;
}

/** The chain from the hips up to each limb. */
export const CHAINS: Record<"leftArm" | "rightArm" | "head", VRMHumanBoneName[]> = {
  leftArm: ["hips", "spine", "chest", "upperChest", "leftShoulder"],
  rightArm: ["hips", "spine", "chest", "upperChest", "rightShoulder"],
  head: ["hips", "spine", "chest", "upperChest", "neck"],
};

/** Down to the forearm, i.e. everything above the wrist. */
export const FOREARM_CHAINS: Record<"left" | "right", VRMHumanBoneName[]> = {
  left: [...CHAINS.leftArm, "leftUpperArm", "leftLowerArm"],
  right: [...CHAINS.rightArm, "rightUpperArm", "rightLowerArm"],
};

/** Frame-rate independent smoothing factor: same feel at 30 or 144 fps. */
export function ease(rate: number, delta: number): number {
  return 1 - Math.exp(-rate * delta);
}
