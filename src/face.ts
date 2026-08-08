import { FaceLandmarker } from "@mediapipe/tasks-vision";
import { visionFileset, MODELS } from "./mediapipe";

export interface FaceResult {
  /** ARKit-style blendshape scores, 0..1, keyed by name. */
  shapes: Map<string, number>;
  /** Row-major 4x4 head transform, or undefined if the model did not solve one. */
  matrix?: number[];
}

/**
 * Face tracking is a second model running beside the pose one. It is worth the
 * cost: the pose model gives four coarse points for the whole head, while this
 * one returns 52 expression scores and a solved head transform.
 */
export class FaceTracker {
  droppedFrames = 0;

  private landmarker: FaceLandmarker | undefined;
  private lastVideoTime = -1;
  private lastTimestamp = 0;
  private last: FaceResult | null = null;

  constructor(private video: HTMLVideoElement) {}

  async init(): Promise<void> {
    const vision = await visionFileset();
    this.landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.face, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
  }

  get ready(): boolean {
    return this.landmarker !== undefined;
  }

  /**
   * Returns this frame's face, or the previous one between camera frames so
   * expressions keep easing rather than stepping.
   */
  update(): FaceResult | null {
    if (!this.landmarker) return null;
    if (this.video.videoWidth === 0) return null;
    if (this.video.currentTime === this.lastVideoTime) return this.last;
    this.lastVideoTime = this.video.currentTime;

    // Same rule as the pose model: a repeated timestamp errors the graph for
    // the rest of the session, so force it forward.
    const timestamp = Math.max(this.lastTimestamp + 1, Math.round(performance.now()));
    this.lastTimestamp = timestamp;

    try {
      const result = this.landmarker.detectForVideo(this.video, timestamp);
      const categories = result.faceBlendshapes?.[0]?.categories;
      if (!categories) {
        this.last = null;
        return null;
      }
      const shapes = new Map<string, number>();
      for (const c of categories) shapes.set(c.categoryName, c.score);
      this.last = { shapes, matrix: result.facialTransformationMatrixes?.[0]?.data };
      return this.last;
    } catch {
      this.droppedFrames++;
      return null;
    }
  }
}
