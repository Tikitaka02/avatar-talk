import { FilesetResolver } from "@mediapipe/tasks-vision";

type WasmFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

/**
 * The JS bundle and the WebAssembly runtime must be the same build, so this is
 * pinned to the exact version in package.json rather than a range. Keep the two
 * in step: a mismatch mostly works, right up until it does not.
 */
export const TASKS_VISION_VERSION = "0.10.35";

const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;

let pending: Promise<WasmFileset> | undefined;

/**
 * The WebAssembly runtime is shared by every task, so it is fetched and
 * initialised once no matter how many landmarkers ask for it.
 */
export function visionFileset(): Promise<WasmFileset> {
  if (!pending) pending = FilesetResolver.forVisionTasks(WASM_ROOT);
  return pending;
}

/** Model files, all of them the lite/float16 builds. */
export const MODELS = {
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
} as const;
