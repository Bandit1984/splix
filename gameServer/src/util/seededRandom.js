/**
 * Creates a deterministic pseudo-random number generator that returns values in [0, 1).
 * Uses Mulberry32 with a 32-bit seed.
 * @param {number} seed
 */
export function createSeededRandom(seed) {
	let t = seed >>> 0;
	return () => {
		t += 0x6D2B79F5;
		let r = Math.imul(t ^ (t >>> 15), 1 | t);
		r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
		return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
	};
}

/**
 * Converts an arbitrary number into a stable 32-bit seed.
 * @param {number} value
 */
export function normalizeSeed(value) {
	if (!Number.isFinite(value)) return 0;
	return (Math.floor(value) >>> 0);
}
