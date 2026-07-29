import {
  FilesetResolver,
  PoseLandmarker,
  type Landmark,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

export interface TrackedPose {
  /** Screen-space landmarks, normalized 0..1 — used to draw the overlay. */
  landmarks: NormalizedLandmark[];
  /** Metric landmarks in a hips-centred space — used to pose the avatar. */
  world: Landmark[];
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

const MIN_VISIBILITY = 0.5;

export class PoseTracker {
  /** Frames inference refused. Non-zero means tracking is degraded, not dead. */
  droppedFrames = 0;
  lastError = "";
  firstError = "";

  private landmarker: PoseLandmarker | undefined;
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
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    this.landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
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
   * Returns this frame's pose, or null if nothing new was processed.
   */
  update(): TrackedPose | null {
    if (!this.landmarker) return null;
    // No camera (denied, or not ready yet): the video is 0x0 and MediaPipe
    // rejects an empty region of interest, so there is nothing to track.
    if (this.video.videoWidth === 0 || this.video.videoHeight === 0) return null;

    // The overlay canvas must match the video's intrinsic resolution so that
    // normalized landmark coordinates line up with the pixels underneath.
    if (
      this.canvas.width !== this.video.videoWidth ||
      this.canvas.height !== this.video.videoHeight
    ) {
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    }

    // Only run inference when the video has a new frame.
    if (this.video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = this.video.currentTime;

    // Inference can reject a frame — a repeated timestamp, a decode hiccup. It
    // throws from inside the render loop, so without this the whole app freezes
    // for good over one bad frame.
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
    if (result.landmarks.length === 0 || result.worldLandmarks.length === 0) return null;

    this.draw(result.landmarks[0]);
    return { landmarks: result.landmarks[0], world: result.worldLandmarks[0] };
  }

  private draw(landmarks: NormalizedLandmark[]): void {
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
}
