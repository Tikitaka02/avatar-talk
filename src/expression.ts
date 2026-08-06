import type { VRM } from "@pixiv/three-vrm";
import type { FaceResult } from "./face";
import { OneEuroFilter, ScalarFilterBank } from "./filter";

/**
 * MediaPipe reports 52 ARKit-style blendshapes; VRM defines its own small set
 * of named expressions. This is the bridge. Several ARKit shapes are per-side
 * where VRM is not, so those are averaged, and a gain compensates for scores
 * that rarely reach 1.0 on a real face.
 */
const MAPPING: { vrm: string; from: string[]; gain: number }[] = [
  { vrm: "blink", from: ["eyeBlinkLeft", "eyeBlinkRight"], gain: 1.1 },
  { vrm: "blinkLeft", from: ["eyeBlinkLeft"], gain: 1.1 },
  { vrm: "blinkRight", from: ["eyeBlinkRight"], gain: 1.1 },
  { vrm: "aa", from: ["jawOpen"], gain: 1.4 },
  { vrm: "ou", from: ["mouthPucker", "mouthFunnel"], gain: 1.3 },
  { vrm: "ih", from: ["mouthStretchLeft", "mouthStretchRight"], gain: 1.5 },
  { vrm: "ee", from: ["mouthSmileLeft", "mouthSmileRight"], gain: 0.8 },
  { vrm: "happy", from: ["mouthSmileLeft", "mouthSmileRight"], gain: 1.6 },
  { vrm: "sad", from: ["mouthFrownLeft", "mouthFrownRight"], gain: 1.4 },
  { vrm: "angry", from: ["browDownLeft", "browDownRight"], gain: 1.2 },
  { vrm: "surprised", from: ["browInnerUp"], gain: 1.2 },
];

/** Below this a score is treated as noise and snapped shut. */
const DEAD_ZONE = 0.06;

export class ExpressionDriver {
  private vrm: VRM | undefined;
  private available = new Set<string>();
  private filters = new ScalarFilterBank(
    // Expressions want a snappier filter than limbs: a blink is 100 ms, and
    // smoothing it like an arm turns it into a slow blink.
    () => new OneEuroFilter(2.5, 0.05)
  );
  private applied = new Map<string, number>();

  bind(vrm: VRM): void {
    this.vrm = vrm;
    this.available = new Set(
      vrm.expressionManager?.expressions.map((e) => e.expressionName) ?? []
    );
    this.applied.clear();
  }

  /** Expression names this avatar actually defines, for the manual controls. */
  get names(): string[] {
    return [...this.available];
  }

  private set(name: string, value: number): void {
    if (!this.available.has(name)) return;
    this.vrm?.expressionManager?.setValue(name, value);
    this.applied.set(name, value);
  }

  apply(face: FaceResult, dt: number): void {
    if (!this.vrm) return;
    const { shapes } = face;

    // Smiling lifts the cheeks, which pushes the eye-blink scores up, so a
    // grinning face reads as a face blinking constantly. Back the blink off by
    // however much the cheeks are raised.
    const squint =
      ((shapes.get("cheekSquintLeft") ?? 0) + (shapes.get("cheekSquintRight") ?? 0)) / 2;

    for (const { vrm: name, from, gain } of MAPPING) {
      if (!this.available.has(name)) continue;

      let sum = 0;
      let n = 0;
      for (const key of from) {
        const v = shapes.get(key);
        if (v !== undefined) {
          sum += v;
          n++;
        }
      }
      if (n === 0) continue;

      let value = (sum / n) * gain;
      if (name.startsWith("blink")) value -= squint * 0.8;

      value = Math.min(1, Math.max(0, value));
      value = this.filters.filter(name, value, dt);
      this.set(name, value < DEAD_ZONE ? 0 : value);
    }
  }

  /** What the mapping produced this frame, for the on-screen readout. */
  get driven(): [string, number][] {
    return [...this.applied.entries()];
  }

  /** Eases every driven expression back to neutral when the face is lost. */
  release(dt: number): void {
    if (!this.vrm || this.applied.size === 0) return;
    const decay = 1 - Math.exp(-6 * dt);
    for (const [name, value] of this.applied) {
      const next = value * (1 - decay);
      this.vrm.expressionManager?.setValue(name, next);
      this.applied.set(name, next);
    }
  }
}
