import {
  FilesetResolver,
  HolisticLandmarker,
  type Landmark,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

/**
 * The JS bundle and the WebAssembly runtime must be the same build, so this is
 * pinned to the exact version in package.json rather than a range.
 */
const TASKS_VISION_VERSION = "0.10.35";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task";

export interface HandStream {
  /** Image-space landmarks, for drawing. */
  landmarks: NormalizedLandmark[];
  /** Metric landmarks, for deriving rotations. */
  world: Landmark[];
}

export interface TrackedFrame {
  /** Screen-space pose landmarks, normalized 0..1 — used to draw the overlay. */
  landmarks: NormalizedLandmark[];
  /** Metric pose landmarks in a hips-centred space — used to pose the avatar. */
  world: Landmark[];
  /** ARKit-style blendshape scores, 0..1, or null when no face was found. */
  faceShapes: Map<string, number> | null;
  /** The person's own left and right hands. */
  leftHand: HandStream | null;
  rightHand: HandStream | null;
}

// Upper body only: landmarks 0–24 (face, arms, torso). Legs (25–32) are
// unreliable from a desk webcam, so we never draw or use them.
const UPPER_BODY_CUTOFF = 25;

// Connection pairs limited to the upper body subset of MediaPipe's pose
// topology: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const UPPER_BODY_CONNECTIONS: [number, number][] = [
  // face outline
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  // torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // left arm + hand
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // right arm + hand
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
];

/** Finger chains for drawing a tracked hand. */
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const MIN_VISIBILITY = 0.5;

/**
 * One model for body, face and hands.
 *
 * Running three separate landmarkers works, but they each re-run their own
 * person detection on the same frame. Holistic shares that stage and returns
 * all three streams together — and it labels the hands itself, so nothing has
 * to infer which hand is which.
 */
export class HolisticTracker {
  /** Frames inference refused. Non-zero means tracking is degraded, not dead. */
  droppedFrames = 0;
  lastError = "";
  firstError = "";

  private landmarker: HolisticLandmarker | undefined;
  private lastVideoTime = -1;
  private lastTimestamp = 0;
  private ctx: CanvasRenderingContext2D;

  constructor(
    private video: HTMLVideoElement,
    private canvas: HTMLCanvasElement
  ) {
    this.ctx = canvas.getContext("2d")!;
  }

  async init(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    this.landmarker = await HolisticLandmarker.createFromOptions(vision, {
      // CPU, not GPU. Holistic's blendshape sub-model uses ops the GPU delegate
      // does not implement (DEQUANTIZE, STRIDED_SLICE), and the graph fails to
      // open — silently, from the caller's point of view. Asking for face
      // expressions therefore costs the GPU path for the whole task.
      baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
      runningMode: "VIDEO",
      outputFaceBlendshapes: true,
    });
  }

  async startWebcam(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
      audio: false,
    });
    this.video.srcObject = stream;
    await new Promise<void>((resolve) => {
      this.video.onloadedmetadata = () => resolve();
    });
  }

  /**
   * Runs inference if the video has a new frame and redraws the overlay.
   * Returns this frame's tracking, or null if nothing new was processed.
   */
  update(): TrackedFrame | null {
    if (!this.landmarker) return null;
    // No camera (denied, or not ready yet): the video is 0x0 and MediaPipe
    // rejects an empty region of interest, so there is nothing to track.
    if (this.video.videoWidth === 0 || this.video.videoHeight === 0) return null;

    // The overlay canvas must match the video's intrinsic resolution so that
    // normalized landmark coordinates line up with the pixels underneath.
    if (this.canvas.width !== this.video.videoWidth || this.canvas.height !== this.video.videoHeight) {
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    }

    // Only run inference when the video has a new frame.
    if (this.video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = this.video.currentTime;

    // MediaPipe demands strictly increasing timestamps and treats a repeat as
    // fatal: the graph errors and every later frame fails, so tracking dies for
    // the session. Two frames can share a millisecond, so force it forward.
    const timestamp = Math.max(this.lastTimestamp + 1, Math.round(performance.now()));
    this.lastTimestamp = timestamp;

    let result;
    try {
      result = this.landmarker.detectForVideo(this.video, timestamp);
    } catch (err) {
      this.droppedFrames++;
      this.lastError = err instanceof Error ? err.message : String(err);
      if (!this.firstError) this.firstError = this.lastError;
      // Clear the overlay too: a stale skeleton left on screen reads as working
      // tracking and hides the fact that nothing is being detected.
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      return null;
    }

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const landmarks = result.poseLandmarks[0];
    const world = result.poseWorldLandmarks[0];
    if (!landmarks || !world) return null;

    this.drawPose(landmarks);

    const hand = (
      lm: NormalizedLandmark[][] | undefined,
      w: Landmark[][] | undefined
    ): HandStream | null => {
      if (!lm?.[0] || !w?.[0]) return null;
      this.drawHand(lm[0]);
      return { landmarks: lm[0], world: w[0] };
    };

    const categories = result.faceBlendshapes?.[0]?.categories;
    let faceShapes: Map<string, number> | null = null;
    if (categories) {
      faceShapes = new Map();
      for (const c of categories) faceShapes.set(c.categoryName, c.score);
    }

    return {
      landmarks,
      world,
      faceShapes,
      leftHand: hand(result.leftHandLandmarks, result.leftHandWorldLandmarks),
      rightHand: hand(result.rightHandLandmarks, result.rightHandWorldLandmarks),
    };
  }

  private drawPose(landmarks: NormalizedLandmark[]): void {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#4ade80";
    for (const [a, b] of UPPER_BODY_CONNECTIONS) {
      const la = landmarks[a];
      const lb = landmarks[b];
      if ((la.visibility ?? 1) < MIN_VISIBILITY) continue;
      if ((lb.visibility ?? 1) < MIN_VISIBILITY) continue;
      ctx.beginPath();
      ctx.moveTo(la.x * w, la.y * h);
      ctx.lineTo(lb.x * w, lb.y * h);
      ctx.stroke();
    }

    ctx.fillStyle = "#fb7185";
    for (let i = 0; i < UPPER_BODY_CUTOFF; i++) {
      const lm = landmarks[i];
      if ((lm.visibility ?? 1) < MIN_VISIBILITY) continue;
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private drawHand(landmarks: NormalizedLandmark[]): void {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#60a5fa";
    for (const [a, b] of HAND_CONNECTIONS) {
      const la = landmarks[a];
      const lb = landmarks[b];
      if (!la || !lb) continue;
      ctx.beginPath();
      ctx.moveTo(la.x * w, la.y * h);
      ctx.lineTo(lb.x * w, lb.y * h);
      ctx.stroke();
    }
    ctx.fillStyle = "#bfdbfe";
    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
