import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { VRM, VRMHumanBoneName } from "@pixiv/three-vrm";
import { OneEuroFilter, ScalarFilterBank } from "./filter";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export interface TrackedHand {
  /** "Left" or "Right", as the model labels the person's own hand. */
  handedness: string;
  landmarks: NormalizedLandmark[];
}

export class HandTracker {
  droppedFrames = 0;

  private landmarker: HandLandmarker | undefined;
  private lastVideoTime = -1;
  private lastTimestamp = 0;
  private last: TrackedHand[] = [];

  constructor(private video: HTMLVideoElement) {}

  async init(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    this.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    });
  }

  get ready(): boolean {
    return this.landmarker !== undefined;
  }

  update(): TrackedHand[] {
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

const MAX_CURL = 1.6;
/** The thumb folds across the palm rather than curling, so it gets less. */
const THUMB_SCALE = 0.55;

function angleBetween(
  a: NormalizedLandmark,
  b: NormalizedLandmark,
  c: NormalizedLandmark
): number {
  const v1 = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const l1 = Math.hypot(v1.x, v1.y, v1.z);
  const l2 = Math.hypot(v2.x, v2.y, v2.z);
  if (l1 === 0 || l2 === 0) return 0;
  const dot = (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / (l1 * l2);
  return Math.acos(Math.min(1, Math.max(-1, dot)));
}

/**
 * Poses the avatar's fingers from tracked hands.
 *
 * Only the bend of each joint is used, not its direction: a finger's curl is
 * the angle between its two adjacent segments, applied around the bone's own
 * curl axis. That deliberately ignores splay, which is far noisier than bend
 * and barely reads on screen.
 */
export class FingerRetargeter {
  private vrm: VRM | undefined;
  private filters = new ScalarFilterBank(() => new OneEuroFilter(1.8, 0.03));
  private current = new Map<VRMHumanBoneName, number>();

  bind(vrm: VRM): void {
    this.vrm = vrm;
    this.current.clear();
  }

  private drive(bone: VRMHumanBoneName, curl: number, side: "left" | "right", dt: number): void {
    const node = this.vrm?.humanoid.getNormalizedBoneNode(bone);
    if (!node) return;
    const smoothed = this.filters.filter(bone, curl, dt);
    // Fingers point along ±x and curl toward the palm, which is a rotation
    // about z — negative on the left hand, positive on the right.
    const signed = side === "left" ? -smoothed : smoothed;
    node.rotation.z = signed;
    this.current.set(bone, signed);
  }

  apply(hands: TrackedHand[], dt: number): boolean {
    if (!this.vrm || hands.length === 0) return false;

    let posed = false;
    for (const hand of hands) {
      // Mirrored, like the arms: the person's right hand drives the avatar's
      // left, so the avatar behaves like a reflection.
      const side = hand.handedness === "Right" ? "left" : "right";
      const prefix = side === "left" ? "left" : "right";
      const lm = hand.landmarks;
      if (lm.length < 21) continue;

      for (const finger of FINGERS) {
        const [a, b, c, d] = finger.chain;
        const isThumb = finger.name === "Thumb";
        const names = isThumb ? THUMB_SEGMENTS : SEGMENTS;
        const scale = isThumb ? THUMB_SCALE : 1;

        // One angle per joint, each between the segments meeting at it.
        const bends = [
          angleBetween(lm[0], lm[a], lm[b]),
          angleBetween(lm[a], lm[b], lm[c]),
          angleBetween(lm[b], lm[c], lm[d]),
        ];

        for (let i = 0; i < 3; i++) {
          const bone = `${prefix}${finger.name}${names[i]}` as VRMHumanBoneName;
          const curl = Math.min(MAX_CURL, Math.max(0, bends[i])) * scale;
          this.drive(bone, curl, side, dt);
        }
      }
      posed = true;
    }
    return posed;
  }

  /** Opens the hands again when tracking is lost. */
  release(dt: number): void {
    if (!this.vrm || this.current.size === 0) return;
    const decay = 1 - Math.exp(-5 * dt);
    for (const [bone, signed] of this.current) {
      const node = this.vrm.humanoid.getNormalizedBoneNode(bone);
      if (!node) continue;
      const next = signed * (1 - decay);
      node.rotation.z = next;
      this.current.set(bone, next);
    }
  }
}
