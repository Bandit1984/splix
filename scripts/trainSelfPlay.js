import { Vec2 } from "renda";
import { Main } from "../gameServer/src/Main.js";

const DIRECTION_VECTORS = {
	right: new Vec2(1, 0),
	down: new Vec2(0, 1),
	left: new Vec2(-1, 0),
	up: new Vec2(0, -1),
};

const ALL_DIRECTIONS = ["right", "down", "left", "up"];

/**
 * @typedef {import("../gameServer/src/gameplay/Player.js").Direction} Direction
 */

/** @param {number} ms */
function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

/** @param {string} name @param {number} fallback */
function parseNumberArg(name, fallback) {
	const prefix = `--${name}=`;
	const rawValue = Deno.args.find((arg) => arg.startsWith(prefix));
	if (!rawValue) return fallback;
	const value = Number(rawValue.slice(prefix.length));
	if (!Number.isFinite(value)) {
		throw new Error(`Invalid numeric argument for --${name}: ${rawValue}`);
	}
	return value;
}

/** @param {string} name @param {string} fallback */
function parseStringArg(name, fallback) {
	const prefix = `--${name}=`;
	const rawValue = Deno.args.find((arg) => arg.startsWith(prefix));
	if (!rawValue) return fallback;
	return rawValue.slice(prefix.length);
}

/**
 * Creates a permissive fake connection with no-op message handlers.
 * Player/Game logic calls many send* methods, this proxy absorbs all of them.
 * @param {() => void} onClose
 * @returns {any}
 */
function createHeadlessConnection(onClose) {
	const connectionState = {
		protocolVersion: 1,
		plusSkinsAllowed: true,
		closed: false,
		close() {
			if (connectionState.closed) return;
			connectionState.closed = true;
			onClose();
		},
		send() {},
	};
	/** @type {any} */
	const proxy = new Proxy(connectionState, {
		get(target, prop) {
			if (prop in target) return target[prop];
			if (typeof prop === "string" && prop.startsWith("send")) {
				return () => {};
			}
			return undefined;
		},
	});
	return proxy;
}

class HeuristicBot {
	/**
	 * @param {import("../gameServer/src/gameplay/Player.js").Player} player
	 * @param {import("../gameServer/src/gameplay/Game.js").Game} game
	 */
	constructor(player, game) {
		this.player = player;
		this.game = game;
		this.lastTurnTime = 0;
	}

	/**
	 * @param {number} now
	 */
	step(now) {
		if (this.player.dead || this.player.permanentlyDead) return;
		if (now - this.lastTurnTime < 100) return;
		this.lastTurnTime = now;

		const currentPosition = this.player.getPosition();
		const currentDirection = this.player.currentDirection;
		let bestDirection = currentDirection == "paused" ? "up" : currentDirection;
		let bestScore = -Infinity;

		for (const direction of ALL_DIRECTIONS) {
			const directionTyped = /** @type {Exclude<import("../gameServer/src/gameplay/Player.js").Direction, "paused">} */ (direction);
			const score = this.#scoreDirection(currentPosition, currentDirection, directionTyped);
			if (score > bestScore) {
				bestScore = score;
				bestDirection = directionTyped;
			}
		}

		this.player.clientPosUpdateRequested(bestDirection, currentPosition);
	}

	/**
	 * @param {Vec2} pos
	 * @param {import("../gameServer/src/gameplay/Player.js").Direction} currentDirection
	 * @param {Exclude<import("../gameServer/src/gameplay/Player.js").Direction, "paused">} nextDirection
	 * @returns {number}
	 */
	#scoreDirection(pos, currentDirection, nextDirection) {
		const nextPos = pos.clone().add(DIRECTION_VECTORS[nextDirection]);

		if (nextPos.x <= 0 || nextPos.y <= 0 || nextPos.x >= this.game.arena.width - 1 || nextPos.y >= this.game.arena.height - 1) {
			return -10_000;
		}

		const tileValue = this.game.arena.getTileValue(nextPos);
		if (tileValue === -1) return -10_000;

		let score = Math.random() * 0.2;
		const distanceToEdge = Math.min(
			nextPos.x,
			nextPos.y,
			this.game.arena.width - 1 - nextPos.x,
			this.game.arena.height - 1 - nextPos.y,
		);
		score += Math.min(distanceToEdge, 8) * 0.05;

		if (tileValue === this.player.id) {
			score += 0.6;
		} else if (tileValue === 0) {
			score += 1.0;
		} else {
			score += 0.2;
		}

		if (
			(currentDirection == "left" && nextDirection == "right") ||
			(currentDirection == "right" && nextDirection == "left") ||
			(currentDirection == "up" && nextDirection == "down") ||
			(currentDirection == "down" && nextDirection == "up")
		) {
			score -= 0.8;
		}
		return score;
	}
}

/**
 * @param {Object} options
 * @param {number} options.botCount
 * @param {number} options.matchDurationMs
 * @param {number} options.arenaWidth
 * @param {number} options.arenaHeight
 * @param {"default" | "drawing" | "arena"} options.gameMode
 */
async function runHeadlessMatch({
	botCount,
	matchDurationMs,
	arenaWidth,
	arenaHeight,
	gameMode,
}) {
	const main = new Main({
		arenaWidth,
		arenaHeight,
		gameMode,
	});

	/** @type {{player: import("../gameServer/src/gameplay/Player.js").Player, removed: boolean}[]} */
	const botEntries = [];
	for (let i = 0; i < botCount; i++) {
		/** @type {any} */
		let botEntry = null;
		const connection = createHeadlessConnection(() => {
			if (!botEntry || botEntry.removed) return;
			botEntry.removed = true;
			main.game.removePlayer(botEntry.player);
		});
		const player = main.game.createPlayer(/** @type {any} */ (connection), {
			name: `bot-${i + 1}`,
			isSpectator: false,
		});
		botEntry = {
			player,
			removed: false,
		};
		botEntries.push(botEntry);
	}

	const bots = botEntries.map((entry) => new HeuristicBot(entry.player, main.game));

	const start = performance.now();
	while (performance.now() - start < matchDurationMs) {
		const now = main.applicationLoop.now;
		for (const bot of bots) {
			bot.step(now);
		}

		const livingCount = botEntries.filter((entry) => !entry.removed && !entry.player.permanentlyDead).length;
		if (livingCount <= 1) break;
		await sleep(25);
	}

	for (const entry of botEntries) {
		if (!entry.removed) {
			entry.removed = true;
			main.game.removePlayer(entry.player);
		}
	}
	main.stop();

	const scores = botEntries.map((entry) => ({
		name: entry.player.name,
		totalScore: entry.player.getTotalScore(),
		totalKills: entry.player.getTotalKill(),
		permanentlyDead: entry.player.permanentlyDead,
	}));
	scores.sort((a, b) => b.totalScore - a.totalScore);
	return scores;
}

const episodes = parseNumberArg("episodes", 10);
const botCount = parseNumberArg("bots", 8);
const matchDurationMs = parseNumberArg("duration-ms", 20_000);
const arenaSize = parseNumberArg("arena-size", 80);
const gameMode = parseStringArg("game-mode", "default");

if (episodes <= 0) throw new Error("--episodes must be >= 1");
if (botCount <= 1) throw new Error("--bots must be >= 2");

let cumulativeBestScore = 0;
let cumulativeTopKills = 0;

console.log(`Starting headless self-play baseline: episodes=${episodes}, bots=${botCount}, arena=${arenaSize}x${arenaSize}, mode=${gameMode}`);

for (let episode = 1; episode <= episodes; episode++) {
	const results = await runHeadlessMatch({
		botCount,
		matchDurationMs,
		arenaWidth: arenaSize,
		arenaHeight: arenaSize,
		gameMode,
	});
	const winner = results[0];
	cumulativeBestScore += winner?.totalScore || 0;
	cumulativeTopKills += winner?.totalKills || 0;
	console.log(
		`episode ${episode}/${episodes}: winner=${winner?.name || "n/a"} score=${winner?.totalScore || 0} kills=${winner?.totalKills || 0}`,
	);
}

console.log("Finished self-play baseline run.");
console.log(`avg winner score=${(cumulativeBestScore / episodes).toFixed(2)} avg winner kills=${(cumulativeTopKills / episodes).toFixed(2)}`);

Deno.exit(0);
