import * as THREE from "three";
import type { VRM, VRMHumanBoneName } from "@pixiv/three-vrm";
import { HandLandmarker, type Landmark, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import { visionFileset, MODELS } from "./mediapipe";
import { OneEuroFilter, ScalarFilterBank } from "./filter";
import { toAvatarSpace, chainWorldQuaternion, FOREARM_CHAINS, ease } from "./space";

export interface HandStream {
  /** "Left" or "Right", as the model labels the person's own hand. */
  handedness: string;
  /** Image-space landmarks, for finger bend. */
  landmarks: NormalizedLandmark[];
  /** Metric landmarks, for solving the wrist's orientation. */
  world: Landmark[];
}

export class HandTracker {
  droppedFrames = 0;

  private landmarker: HandLandmarker | undefined;
  private lastVideoTime = -1;
  private lastTimestamp = 0;
  private last: HandStream[] = [];

  constructor(private video: HTMLVideoElement) {}

  async init(): Promise<void> {
    const vision = await visionFileset();
    this.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.hand, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    });
  }

  get ready(): boolean {
    return this.landmarker !== undefined;
  }

  /** Returns the previous result between camera frames so easing continues. */
  update(): HandStream[] {
    if (!this.landmarker || this.video.videoWidth === 0) return [];
    if (this.video.currentTime === this.lastVideoTime) return this.last;
    this.lastVideoTime = this.video.currentTime;

    const timestamp = Math.max(this.lastTimestamp + 1, Math.round(performance.now()));
    this.lastTimestamp = timestamp;

    try {
      const result = this.landmarker.detectForVideo(this.video, timestamp);
      this.last = result.landmarks.map((landmarks, i) => ({
        handedness: result.handedness[i]?.[0]?.categoryName ?? "Right",
        landmarks,
        world: result.worldLandmarks[i] ?? [],
      }));
      return this.last;
    } catch {
      this.droppedFrames++;
      return [];
    }
  }
}

/** MediaPipe hand landmark indices, four joints per finger from knuckle to tip. */
const FINGERS: { name: string; chain: [number, number, number, number] }[] = [
  { name: "Thumb", chain: [1, 2, 3, 4] },
  { name: "Index", chain: [5, 6, 7, 8] },
  { name: "Middle", chain: [9, 10, 11, 12] },
  { name: "Ring", chain: [13, 14, 15, 16] },
  { name: "Little", chain: [17, 18, 19, 20] },
];

/** VRM names the three finger bones differently for the thumb. */
const SEGMENTS = ["Proximal", "Intermediate", "Distal"] as const;
const THUMB_SEGMENTS = ["Metacarpal", "Proximal", "Distal"] as const;

const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;

const MAX_CURL = 1.6;
/** The thumb folds across the palm rather than curling, so it gets less. */
const THUMB_SCALE = 0.55;

/**
 * How much of the solved wrist rotation to keep. Palm orientation leans on
 * depth, the weakest axis a single camera has, so easing off a little keeps a
 * mis-read palm from wrenching the hand.
 */
const WRIST_DAMPING = 0.85;

/**
 * Ceiling on the wrist rotation, generous on purpose. Holding your palms at the
 * camera is already ~90° away from the rest pose, where the palms face down, so
 * a tight cap rejects ordinary poses. Rotations past this are scaled back to it
 * rather than dropped: discarding a frame makes the hand stall and snap, which
 * reads as the wrist not following at all.
 */
const MAX_WRIST_ANGLE = Math.PI * 0.75;

/**
 * Rest orientation of each hand in the normalized rig. VRM's T-pose has the
 * arms along ±x with the palms facing down and the thumbs forward, so the
 * knuckles run from index at +z to little finger at -z on both hands.
 */
const REST_BASIS = {
  left: { forward: new THREE.Vector3(1, 0, 0), side: new THREE.Vector3(0, 0, -1) },
  right: { forward: new THREE.Vector3(-1, 0, 0), side: new THREE.Vector3(0, 0, -1) },
};

function angleBetween(a: Landmark, b: Landmark, c: Landmark): number {
  const v1x = b.x - a.x, v1y = b.y - a.y, v1z = b.z - a.z;
  const v2x = c.x - b.x, v2y = c.y - b.y, v2z = c.z - b.z;
  const l1 = Math.hypot(v1x, v1y, v1z);
  const l2 = Math.hypot(v2x, v2y, v2z);
  if (l1 === 0 || l2 === 0) return 0;
  const dot = (v1x * v2x + v1y * v2y + v1z * v2z) / (l1 * l2);
  return Math.acos(Math.min(1, Math.max(-1, dot)));
}

/** Builds an orthonormal frame as a matrix whose columns are forward, side, normal. */
function basisMatrix(
  forward: THREE.Vector3,
  side: THREE.Vector3,
  out: THREE.Matrix4,
  scratch: { n: THREE.Vector3; s: THREE.Vector3 }
): THREE.Matrix4 {
  const n = scratch.n.crossVectors(forward, side).normalize();
  const s = scratch.s.crossVectors(n, forward).normalize();
  return out.makeBasis(forward, s, n);
}

/**
 * Poses the avatar's hands: the curl of each finger joint, and the rotation of
 * the wrist itself.
 *
 * Finger curl uses only the bend — the angle between the two segments meeting
 * at a joint — applied around the bone's own curl axis. That deliberately
 * ignores splay, which is far noisier than bend and barely reads on screen.
 */
export class FingerRetargeter {
  private vrm: VRM | undefined;
  private filters = new ScalarFilterBank(() => new OneEuroFilter(1.8, 0.03));
  private curls = new Map<VRMHumanBoneName, number>();
  private wristFilters = new ScalarFilterBank(() => new OneEuroFilter(1.4, 0.02));

  private v = {
    a: new THREE.Vector3(),
    b: new THREE.Vector3(),
    c: new THREE.Vector3(),
    forward: new THREE.Vector3(),
    side: new THREE.Vector3(),
    n: new THREE.Vector3(),
    s: new THREE.Vector3(),
  };
  private m = { measured: new THREE.Matrix4(), rest: new THREE.Matrix4() };
  private q = {
    measured: new THREE.Quaternion(),
    rest: new THREE.Quaternion(),
    target: new THREE.Quaternion(),
    parent: new THREE.Quaternion(),
  };

  bind(vrm: VRM): void {
    this.vrm = vrm;
    this.curls.clear();
  }

  private drive(bone: VRMHumanBoneName, curl: number, side: "left" | "right", dt: number): void {
    const node = this.vrm?.humanoid.getNormalizedBoneNode(bone);
    if (!node) return;
    const smoothed = this.filters.filter(bone, curl, dt);
    // Fingers point along ±x and curl toward the palm, which is a rotation
    // about z — negative on the left hand, positive on the right.
    const signed = side === "left" ? -smoothed : smoothed;
    node.rotation.z = signed;
    this.curls.set(bone, signed);
  }

  /**
   * Solves the wrist from the palm itself. Two vectors define a hand's
   * orientation — along the fingers, and across the knuckles — and the rotation
   * that carries the rest pose's pair onto the measured pair is the wrist.
   * This is the roll that a shoulder-to-elbow direction can never recover.
   */
  private driveWrist(hand: HandStream, side: "left" | "right", dt: number): void {
    const vrm = this.vrm;
    const bone: VRMHumanBoneName = side === "left" ? "leftHand" : "rightHand";
    const node = vrm?.humanoid.getNormalizedBoneNode(bone);
    if (!vrm || !node) return;

    const lm = hand.world;
    if (lm.length < 21) return;

    const { a, b, c, forward, side: across, n, s } = this.v;
    // Depth is kept in full here: the palm's facing is mostly a depth signal,
    // and shrinking z would flatten every rotation this is meant to find.
    const wrist = toAvatarSpace(lm[WRIST], a, 1);
    const middle = toAvatarSpace(lm[MIDDLE_MCP], b, 1);
    forward.subVectors(middle, wrist);
    if (forward.lengthSq() < 1e-8) return;
    forward.normalize();

    const index = toAvatarSpace(lm[INDEX_MCP], b, 1);
    const pinky = toAvatarSpace(lm[PINKY_MCP], c, 1);
    across.subVectors(pinky, index);
    if (across.lengthSq() < 1e-8) return;
    across.normalize();

    basisMatrix(forward, across, this.m.measured, { n, s });
    const rest = REST_BASIS[side];
    basisMatrix(rest.forward, rest.side, this.m.rest, { n, s });

    this.q.measured.setFromRotationMatrix(this.m.measured);
    this.q.rest.setFromRotationMatrix(this.m.rest);
    // World rotation carrying rest onto measured.
    this.q.target.copy(this.q.measured).multiply(this.q.rest.invert());

    // Back out the arm's rotation so what is left is the wrist relative to the
    // forearm, which is where the bone's rotation actually lives.
    chainWorldQuaternion(vrm, FOREARM_CHAINS[side], this.q.parent);
    this.q.target.premultiply(this.q.parent.invert());

    const identity = this.q.rest.identity();

    // Scale an over-large rotation back to the ceiling instead of dropping it.
    const angle = 2 * Math.acos(Math.min(1, Math.abs(this.q.target.w)));
    if (angle > MAX_WRIST_ANGLE) {
      this.q.target.slerp(identity, 1 - MAX_WRIST_ANGLE / angle);
    }
    this.q.target.slerp(identity, 1 - WRIST_DAMPING);

    // The whole rotation goes on the hand. Anatomically most of this roll is
    // the forearm, but the arm solve rewrites the forearm every frame by
    // easing toward its own target: a twist added on top would be partly
    // undone, re-added, and settle several times larger than intended. An
    // absolute target on one bone cannot run away like that.
    node.quaternion.slerp(this.q.target, ease(12, dt));
  }

  private poseHand(hand: HandStream, side: "left" | "right", dt: number): void {
    const lm = hand.landmarks;
    if (lm.length < 21) return;
    const prefix = side;

    for (const finger of FINGERS) {
      const [p1, p2, p3, p4] = finger.chain;
      const isThumb = finger.name === "Thumb";
      const names = isThumb ? THUMB_SEGMENTS : SEGMENTS;
      const scale = isThumb ? THUMB_SCALE : 1;

      // One angle per joint, each between the segments meeting at it.
      const bends = [
        angleBetween(lm[WRIST], lm[p1], lm[p2]),
        angleBetween(lm[p1], lm[p2], lm[p3]),
        angleBetween(lm[p2], lm[p3], lm[p4]),
      ];

      for (let i = 0; i < 3; i++) {
        const bone = `${prefix}${finger.name}${names[i]}` as VRMHumanBoneName;
        const curl = Math.min(MAX_CURL, Math.max(0, bends[i])) * scale;
        this.drive(bone, curl, side, dt);
      }
    }

    this.driveWrist(hand, side, dt);
  }

  /**
   * Mirrored, like the arms: the person's right hand drives the avatar's left,
   * so the avatar behaves like a reflection. MediaPipe's handedness is
   * anatomical — "Right" is the person's own right hand.
   */
  apply(hands: HandStream[], dt: number): boolean {
    if (!this.vrm || hands.length === 0) return false;
    let posed = false;
    for (const hand of hands) {
      this.poseHand(hand, hand.handedness === "Right" ? "left" : "right", dt);
      posed = true;
    }
    return posed;
  }

  /** Opens the hands again when tracking is lost. */
  release(dt: number): void {
    if (!this.vrm || this.curls.size === 0) return;
    const decay = ease(5, dt);
    for (const [bone, signed] of this.curls) {
      const node = this.vrm.humanoid.getNormalizedBoneNode(bone);
      if (!node) continue;
      const next = signed * (1 - decay);
      node.rotation.z = next;
      this.curls.set(bone, next);
    }
    for (const side of ["left", "right"] as const) {
      const node = this.vrm.humanoid.getNormalizedBoneNode(
        side === "left" ? "leftHand" : "rightHand"
      );
      if (node) node.quaternion.slerp(this.q.rest.identity(), decay);
    }
  }
}
