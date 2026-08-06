/**
 * One Euro filter — the standard answer to noisy interactive signals.
 *
 * A plain low-pass filter forces a trade: smooth enough to kill jitter means
 * sluggish enough to feel laggy. One Euro escapes it by making the cutoff
 * depend on speed: when the value is barely moving it filters hard, and when
 * you move fast it lets the signal through. Slow hands stop shaking, quick
 * hands stay responsive.
 *
 * Casiez, Roussel & Vogel, CHI 2012.
 */
export class OneEuroFilter {
  private previous = 0;
  private derivative = 0;
  private started = false;

  constructor(
    /** Cutoff at rest, in Hz. Lower is steadier and laggier. */
    private minCutoff = 1.2,
    /** How much speed raises the cutoff. Higher reacts faster to fast motion. */
    private beta = 0.02,
    /** Cutoff for the speed estimate itself. */
    private dCutoff = 1.0
  ) {}

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(value: number, dt: number): number {
    if (!this.started) {
      this.started = true;
      this.previous = value;
      return value;
    }
    if (dt <= 0) return this.previous;

    // Speed, itself low-passed so a single noisy sample cannot open the gate.
    const rawDerivative = (value - this.previous) / dt;
    const aD = OneEuroFilter.alpha(this.dCutoff, dt);
    this.derivative += aD * (rawDerivative - this.derivative);

    const cutoff = this.minCutoff + this.beta * Math.abs(this.derivative);
    const a = OneEuroFilter.alpha(cutoff, dt);
    this.previous += a * (value - this.previous);
    return this.previous;
  }

  reset(): void {
    this.started = false;
    this.derivative = 0;
  }
}

/** A bank of One Euro filters, one per named scalar. */
export class ScalarFilterBank {
  private filters = new Map<string, OneEuroFilter>();

  constructor(private make: () => OneEuroFilter) {}

  filter(key: string, value: number, dt: number): number {
    let f = this.filters.get(key);
    if (!f) {
      f = this.make();
      this.filters.set(key, f);
    }
    return f.filter(value, dt);
  }
}

export interface Point3 {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/**
 * Smooths a whole landmark stream in place-free fashion: three filters per
 * landmark. Filtering here, before any retargeting, is deliberate — jitter is
 * a property of the measurement, and cleaning it at the source means every
 * angle derived from it inherits the fix.
 */
export class LandmarkFilter {
  private axes: OneEuroFilter[][] = [];

  constructor(private make: () => OneEuroFilter) {}

  apply<T extends Point3>(points: T[], dt: number): T[] {
    for (let i = 0; i < points.length; i++) {
      let f = this.axes[i];
      if (!f) {
        f = [this.make(), this.make(), this.make()];
        this.axes[i] = f;
      }
      const p = points[i];
      p.x = f[0].filter(p.x, dt);
      p.y = f[1].filter(p.y, dt);
      p.z = f[2].filter(p.z, dt);
    }
    return points;
  }

  reset(): void {
    for (const trio of this.axes) for (const f of trio) f?.reset();
  }
}
