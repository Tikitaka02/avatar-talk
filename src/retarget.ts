import * as THREE from "three";
import type { VRM } from "@pixiv/three-vrm";
import type { VRMHumanBoneName } from "@pixiv/three-vrm";
import type { Landmark } from "@mediapipe/tasks-vision";

/** Upper-body landmark indices from MediaPipe's pose topology. */
const LM = {
  nose: 0,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
} as const;

const MIN_VISIBILITY = 0.6;

/** How fast tracked rotations chase their target, per second. */
const FOLLOW_RATE = 14;
/** How fast bones drift home when tracking is lost. */
const RELEASE_RATE = 4;

/** Torso motion is damped: raw shoulder noise looks like a twitch, not a lean. */
const TORSO_DAMPING = 0.55;

/**
 * How far to trust MediaPipe's depth. A single camera cannot measure z, so the
 * model infers it — and it drifts badly: a symmetric pose can report one elbow
 * twice as far forward as the other. Shrinking z pulls limbs back toward the
 * silhouette the camera actually saw, which is the part it gets right.
 */
const Z_TRUST = 0.45;

/**
 * The chain from the hips up to each limb. In the normalized humanoid rig every
 * bone rests axis-aligned with the world, so multiplying the local rotations
 * along a chain gives that joint's world rotation without touching matrices.
 */
const CHAINS: Record<"leftArm" | "rightArm" | "head", VRMHumanBoneName[]> = {
  leftArm: ["hips", "spine", "chest", "upperChest", "leftShoulder"],
  rightArm: ["hips", "spine", "chest", "upperChest", "rightShoulder"],
  head: ["hips", "spine", "chest", "upperChest", "neck"],
};

/** Rest directions of each bone in the normalized T-pose rig. */
const REST_DIR = {
  leftArm: new THREE.Vector3(1, 0, 0),
  rightArm: new THREE.Vector3(-1, 0, 0),
  up: new THREE.Vector3(0, 1, 0),
  forward: new THREE.Vector3(0, 0, 1),
};

const BONES: VRMHumanBoneName[] = [
  "spine",
  "chest",
  "upperChest",
  "neck",
  "head",
  "leftUpperArm",
  "leftLowerArm",
  "rightUpperArm",
  "rightLowerArm",
];

export class PoseRetargeter {
  /** Pose the avatar returns to when tracking stops. */
  private rest = new Map<VRMHumanBoneName, THREE.Quaternion>();
  private vrm: VRM | undefined;

  // Scratch objects, reused every frame to keep the loop allocation-free.
  private v = {
    a: new THREE.Vector3(),
    b: new THREE.Vector3(),
    c: new THREE.Vector3(),
    d: new THREE.Vector3(),
    shoulderMid: new THREE.Vector3(),
    hipMid: new THREE.Vector3(),
    earMid: new THREE.Vector3(),
  };
  private q = {
    target: new THREE.Quaternion(),
    parent: new THREE.Quaternion(),
    scratch: new THREE.Quaternion(),
  };

  /** Records the loaded avatar's rest pose so tracking can be released to it. */
  bind(vrm: VRM): void {
    this.vrm = vrm;
    this.rest.clear();
    for (const name of BONES) {
      const node = vrm.humanoid.getNormalizedBoneNode(name);
      if (node) this.rest.set(name, node.quaternion.clone());
    }
  }

  /**
   * MediaPipe world landmarks are metric, with x to the image right, y down and
   * z growing away from the camera. VRM 1.0 is y up and +z forward, so y and z
   * flip. x flips too, which mirrors the pose: raise your right hand and the
   * avatar raises the hand on the same side of the screen, like a mirror. That
   * mirroring is also why right-side landmarks drive the avatar's left bones.
   */
  private toAvatarSpace(lm: Landmark, out: THREE.Vector3, zTrust = Z_TRUST): THREE.Vector3 {
    return out.set(-lm.x, -lm.y, -lm.z * zTrust);
  }

  private visible(world: Landmark[], ...indices: number[]): boolean {
    return indices.every((i) => (world[i]?.visibility ?? 1) >= MIN_VISIBILITY);
  }

  /** World rotation of a joint, from the local rotations above it. */
  private chainQuaternion(chain: VRMHumanBoneName[], out: THREE.Quaternion): THREE.Quaternion {
    out.identity();
    for (const name of chain) {
      const node = this.vrm?.humanoid.getNormalizedBoneNode(name);
      if (node) out.multiply(node.quaternion);
    }
    return out;
  }

  private ease(rate: number, delta: number): number {
    // Frame-rate independent smoothing: same feel at 30 or 144 fps.
    return 1 - Math.exp(-rate * delta);
  }

  private driveBone(
    name: VRMHumanBoneName,
    target: THREE.Quaternion,
    alpha: number
  ): void {
    const node = this.vrm?.humanoid.getNormalizedBoneNode(name);
    if (node) node.quaternion.slerp(target, alpha);
  }

  /** Eases every tracked bone back to the avatar's rest pose. */
  releaseToRest(delta: number): void {
    if (!this.vrm) return;
    const alpha = this.ease(RELEASE_RATE, delta);
    for (const name of BONES) {
      const rest = this.rest.get(name);
      if (rest) this.driveBone(name, rest, alpha);
    }
  }

  /**
   * Maps one frame of pose landmarks onto the avatar's bones.
   * Returns false when the pose is too uncertain to use.
   */
  apply(world: Landmark[], delta: number): boolean {
    const vrm = this.vrm;
    if (!vrm || world.length === 0) return false;
    if (!this.visible(world, LM.leftShoulder, LM.rightShoulder)) return false;

    const alpha = this.ease(FOLLOW_RATE, delta);
    const { a, b, c, d, shoulderMid, hipMid, earMid } = this.v;
    const { target, parent, scratch } = this.q;

    // --- torso -------------------------------------------------------------
    // Mirrored: the person's right shoulder becomes the avatar's left.
    const avatarLeftShoulder = this.toAvatarSpace(world[LM.rightShoulder], a);
    const avatarRightShoulder = this.toAvatarSpace(world[LM.leftShoulder], b);
    shoulderMid.addVectors(avatarLeftShoulder, avatarRightShoulder).multiplyScalar(0.5);

    if (this.visible(world, LM.leftHip, LM.rightHip)) {
      hipMid
        .addVectors(
          this.toAvatarSpace(world[LM.rightHip], c),
          this.toAvatarSpace(world[LM.leftHip], d)
        )
        .multiplyScalar(0.5);
      // Spine leans and pitches to follow the hips-to-shoulders axis.
      c.subVectors(shoulderMid, hipMid).normalize();
      target.setFromUnitVectors(REST_DIR.up, c);
      scratch.identity().slerp(target, TORSO_DAMPING);
      this.driveBone("spine", scratch, alpha);
    }

    // Shoulder line twist turns the chest. A vector rotated θ about y lands at
    // (cos θ, 0, -sin θ), so the yaw back out of it is atan2(-z, x).
    c.subVectors(avatarLeftShoulder, avatarRightShoulder).normalize();
    const twist = Math.atan2(-c.z, c.x);
    scratch.setFromAxisAngle(REST_DIR.up, twist * TORSO_DAMPING);
    this.driveBone(vrm.humanoid.getNormalizedBoneNode("upperChest") ? "upperChest" : "chest", scratch, alpha);

    // --- arms --------------------------------------------------------------
    this.driveArm(world, "left", LM.rightShoulder, LM.rightElbow, LM.rightWrist, delta);
    this.driveArm(world, "right", LM.leftShoulder, LM.leftElbow, LM.leftWrist, delta);

    // --- head --------------------------------------------------------------
    if (this.visible(world, LM.leftEar, LM.rightEar, LM.nose)) {
      // Head keeps full depth: ears-to-nose points almost straight along z, so
      // shrinking it here would exaggerate every turn instead of steadying it.
      earMid
        .addVectors(
          this.toAvatarSpace(world[LM.leftEar], a, 1),
          this.toAvatarSpace(world[LM.rightEar], b, 1)
        )
        .multiplyScalar(0.5);
      // The nose sits in front of the ears, so that vector is where you look.
      c.subVectors(this.toAvatarSpace(world[LM.nose], d, 1), earMid).normalize();
      target.setFromUnitVectors(REST_DIR.forward, c);
      this.chainQuaternion(CHAINS.head, parent);
      target.premultiply(parent.invert());
      this.driveBone("head", target, alpha);
    }

    return true;
  }

  private driveArm(
    world: Landmark[],
    side: "left" | "right",
    shoulderIdx: number,
    elbowIdx: number,
    wristIdx: number,
    delta: number
  ): void {
    const upper: VRMHumanBoneName = side === "left" ? "leftUpperArm" : "rightUpperArm";
    const lower: VRMHumanBoneName = side === "left" ? "leftLowerArm" : "rightLowerArm";
    const restDir = side === "left" ? REST_DIR.leftArm : REST_DIR.rightArm;
    const alpha = this.ease(FOLLOW_RATE, delta);
    const { a, b, c } = this.v;
    const { target, parent } = this.q;

    if (!this.visible(world, shoulderIdx, elbowIdx)) {
      // Arm out of frame: let it fall home rather than freeze mid-air.
      const release = this.ease(RELEASE_RATE, delta);
      const rest = this.rest.get(upper);
      const restLower = this.rest.get(lower);
      if (rest) this.driveBone(upper, rest, release);
      if (restLower) this.driveBone(lower, restLower, release);
      return;
    }

    const shoulder = this.toAvatarSpace(world[shoulderIdx], a);
    const elbow = this.toAvatarSpace(world[elbowIdx], b);

    // Upper arm: rotate its rest direction onto shoulder → elbow.
    c.subVectors(elbow, shoulder).normalize();
    target.setFromUnitVectors(restDir, c);
    this.chainQuaternion(side === "left" ? CHAINS.leftArm : CHAINS.rightArm, parent);
    target.premultiply(parent.invert());
    this.driveBone(upper, target, alpha);
    // chainQuaternion inverted `parent` in place; rebuild it for the forearm.
    this.chainQuaternion(side === "left" ? CHAINS.leftArm : CHAINS.rightArm, parent);

    if (!this.visible(world, wristIdx)) return;

    // Forearm: same trick, but its parent now includes the upper arm we just
    // set, so the elbow bend comes out relative to the arm rather than the world.
    const wrist = this.toAvatarSpace(world[wristIdx], a);
    c.subVectors(wrist, elbow).normalize();
    target.setFromUnitVectors(restDir, c);
    parent.multiply(this.vrm!.humanoid.getNormalizedBoneNode(upper)!.quaternion);
    target.premultiply(parent.invert());
    this.driveBone(lower, target, alpha);
  }
}
