// @ts-nocheck
import { Vec2 } from "renda";
import { init as initGameServer } from "../mainInstance.js";
import { createSeededRandom, normalizeSeed } from "../util/seededRandom.js";

const DIRECTION_BY_ACTION = ["right", "down", "left", "up", "paused"];
const OPPOSITE_DIRECTIONS = {
	right: "left",
	left: "right",
	up: "down",
	down: "up",
	paused: "paused",
};

/** @typedef {import("../gameplay/Player.js").Direction} Direction */

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

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
	return new Proxy(connectionState, {
		get(target, prop) {
			if (prop in target) return target[prop];
			if (typeof prop == "string" && prop.startsWith("send")) {
				return () => {};
			}
			return undefined;
		},
	});
}

class HeuristicBotController {
	constructor(player, game, randomFn = Math.random) {
		this.player = player;
		this.game = game;
		this.random = randomFn;
		this.lastDecisionTime = 0;
	}

	step(now) {
		if (this.player.dead || this.player.permanentlyDead) return;
		if (now - this.lastDecisionTime < 100) return;
		this.lastDecisionTime = now;

		const currentPosition = this.player.getPosition();
		const currentDirection = this.player.currentDirection;
		let bestDirection = currentDirection == "paused" ? "up" : currentDirection;
		let bestScore = -Infinity;

		for (const direction of /** @type {const} */ (["right", "down", "left", "up"])) {
			const score = this.#scoreDirection(currentPosition, currentDirection, direction);
			if (score > bestScore) {
				bestScore = score;
				bestDirection = direction;
			}
		}
		this.player.clientPosUpdateRequested(bestDirection, currentPosition);
	}

	#scoreDirection(currentPosition, currentDirection, direction) {
		const vector = direction == "right"
			? new Vec2(1, 0)
			: direction == "down"
			? new Vec2(0, 1)
			: direction == "left"
			? new Vec2(-1, 0)
			: new Vec2(0, -1);
		const nextPos = currentPosition.clone().add(vector);
		if (nextPos.x <= 0 || nextPos.y <= 0 || nextPos.x >= this.game.arena.width - 1 || nextPos.y >= this.game.arena.height - 1) {
			return -10_000;
		}
		const tileValue = this.game.arena.getTileValue(nextPos);
		if (tileValue == -1) return -10_000;

		let score = this.random() * 0.2;
		if (tileValue == this.player.id) {
			score += 0.4;
		} else if (tileValue == 0) {
			score += 1.0;
		} else {
			score += 0.2;
		}

		const oppositeDirection = OPPOSITE_DIRECTIONS[currentDirection];
		if (direction == oppositeDirection) {
			score -= 0.5;
		}
		return score;
	}
}

class BridgeSession {
	constructor(envId, options) {
		this.envId = envId;
		this.options = options;
		this.seed = Number.isInteger(options.seed) ? normalizeSeed(options.seed) : null;
		this.randomFn = this.seed == null ? Math.random : createSeededRandom(this.seed);
		this.maxSteps = options.maxSteps;
		this.decisionIntervalMs = options.decisionIntervalMs;
		this.obsRadius = options.obsRadius;
		this.main = null;
		this.controlledEntry = null;
		this.opponentEntries = [];
		this.opponentControllers = [];
		this.opponentInterval = null;
		this.stepCount = 0;
		this.lastScore = 0;
		this.lastKills = 0;
		this.terminalEventLogged = false;
		this.closed = false;
	}

	reset() {
		this.close();
		this.closed = false;
		this.stepCount = 0;
		this.terminalEventLogged = false;
		this.randomFn = this.seed == null ? Math.random : createSeededRandom(this.seed);
		this.main = initGameServer({
			arenaWidth: this.options.arenaWidth,
			arenaHeight: this.options.arenaHeight,
			pitWidth: 16,
			pitHeight: 16,
			gameMode: this.options.gameMode,
			seed: this.seed == null ? undefined : this.seed,
		});

		this.controlledEntry = this.#createPlayer("agent", false);
		this.opponentEntries = [];
		for (let i = 0; i < this.options.opponentCount; i++) {
			this.opponentEntries.push(this.#createPlayer(`opponent-${i + 1}`, true));
		}
		this.opponentControllers = this.opponentEntries.map((entry) => new HeuristicBotController(entry.player, this.main.game, this.randomFn));
		this.opponentInterval = setInterval(() => {
			const now = this.main?.applicationLoop.now || 0;
			for (const controller of this.opponentControllers) {
				controller.step(now);
			}
		}, 40);

		this.lastScore = this.controlledEntry.player.getTotalScore();
		this.lastKills = this.controlledEntry.player.getTotalKill();
		return this.observe();
	}

	#createPlayer(name, opponent) {
		/** @type {any} */
		let entry = null;
		const connection = createHeadlessConnection(() => {
			if (!entry || entry.removed || !this.main) return;
			entry.removed = true;
			this.main.game.removePlayer(entry.player);
		});
		const player = this.main.game.createPlayer(/** @type {any} */ (connection), {
			name,
			isSpectator: false,
		});
		entry = {
			player,
			connection,
			removed: false,
			opponent,
		};
		return entry;
	}

	observe() {
		if (!this.main || !this.controlledEntry || this.controlledEntry.removed) {
			throw new Error("Environment is not initialized. Call reset first.");
		}
		const player = this.controlledEntry.player;
		const deathState = player.getDeathState();
		const position = player.getPosition();
		const localTiles = [];
		for (let y = -this.obsRadius; y <= this.obsRadius; y++) {
			for (let x = -this.obsRadius; x <= this.obsRadius; x++) {
				const tilePos = new Vec2(position.x + x, position.y + y);
				let tileValue = -1;
				if (tilePos.x >= 0 && tilePos.y >= 0 && tilePos.x < this.main.game.arena.width && tilePos.y < this.main.game.arena.height) {
					tileValue = this.main.game.arena.getTileValue(tilePos);
				}
				if (tileValue == player.id) {
					localTiles.push(1);
				} else if (tileValue == 0) {
					localTiles.push(0);
				} else if (tileValue == -1) {
					localTiles.push(-1);
				} else {
					localTiles.push(2);
				}
			}
		}
		return {
			env_id: this.envId,
			step: this.stepCount,
			arena: {
				width: this.main.game.arena.width,
				height: this.main.game.arena.height,
			},
			player: {
				x: position.x,
				y: position.y,
				direction: player.currentDirection,
				score: player.getTotalScore(),
				kills: player.getTotalKill(),
				dead: player.dead,
				permanently_dead: player.permanentlyDead,
				death_type: deathState?.type || null,
				death_killer_name: deathState?.killerName || "",
			},
			local_tiles: localTiles,
			seed: this.seed,
		};
	}

	#getTerminalEvent(observation, truncated) {
		const deathType = observation.player.death_type;
		if (deathType) {
			const causeBucket = deathType == "arena-bounds"
				? "wall"
				: deathType == "self"
				? "self"
				: deathType == "player"
				? "opponent"
				: "unknown";
			return {
				terminal_type: "death",
				cause_bucket: causeBucket,
				death_type: deathType,
				killer_name: observation.player.death_killer_name,
			};
		}
		if (truncated) {
			return {
				terminal_type: "truncated",
				cause_bucket: "truncated",
				death_type: null,
				killer_name: "",
			};
		}
		return null;
	}

	#emitTerminalLog(terminalEvent, observation, reward, info) {
		if (this.terminalEventLogged || !terminalEvent) return;
		this.terminalEventLogged = true;
		console.log(JSON.stringify({
			type: "ai_bridge_episode_terminal",
			env_id: this.envId,
			seed: this.seed,
			step: this.stepCount,
			reward,
			score: observation.player.score,
			kills: observation.player.kills,
			...terminalEvent,
			info,
		}));
	}

	async step(action) {
		if (!this.main || !this.controlledEntry || this.controlledEntry.removed) {
			throw new Error("Environment is not initialized. Call reset first.");
		}
		if (this.closed) {
			throw new Error("Environment is closed.");
		}
		if (!Number.isInteger(action) || action < 0 || action >= DIRECTION_BY_ACTION.length) {
			throw new Error("Action must be an integer between 0 and 4.");
		}

		const controlledPlayer = this.controlledEntry.player;
		const actionDirection = DIRECTION_BY_ACTION[action];
		controlledPlayer.clientPosUpdateRequested(actionDirection, controlledPlayer.getPosition());

		await sleep(this.decisionIntervalMs);
		this.stepCount++;

		const observation = this.observe();
		const score = controlledPlayer.getTotalScore();
		const kills = controlledPlayer.getTotalKill();
		const scoreDelta = score - this.lastScore;
		const killsDelta = kills - this.lastKills;
		this.lastScore = score;
		this.lastKills = kills;

		let reward = scoreDelta * this.options.rewardWeights.score + killsDelta * this.options.rewardWeights.kill;
		if (observation.player.dead || observation.player.permanently_dead) {
			reward += this.options.rewardWeights.death;
		}

		const done = observation.player.dead || observation.player.permanently_dead;
		const truncated = this.stepCount >= this.maxSteps;
		if (truncated && !done) {
			reward += this.options.rewardWeights.truncate;
		}
		const terminalEvent = this.#getTerminalEvent(observation, truncated && !done);
		this.#emitTerminalLog(terminalEvent, observation, reward, {
			score_delta: scoreDelta,
			kills_delta: killsDelta,
			step_count: this.stepCount,
		});

		return {
			observation,
			reward,
			done,
			truncated,
			info: {
				score_delta: scoreDelta,
				kills_delta: killsDelta,
				step_count: this.stepCount,
				terminal_event: terminalEvent,
			},
		};
	}

	close() {
		if (this.closed) return;
		this.closed = true;
		if (this.opponentInterval != null) {
			clearInterval(this.opponentInterval);
			this.opponentInterval = null;
		}
		if (this.main) {
			for (const entry of [this.controlledEntry, ...this.opponentEntries]) {
				if (entry && !entry.removed) {
					entry.removed = true;
					entry.connection.close();
				}
			}
			this.main.stop();
			this.main = null;
		}
		this.controlledEntry = null;
		this.opponentEntries = [];
		this.opponentControllers = [];
	}
}

export class AiBridgeServer {
	constructor() {
		this.sessions = new Map();
		this.nextEnvId = 1;
	}

	/**
	 * @param {Request} request
	 */
	handleRequest(request) {
		if (request.method != "GET") {
			return new Response("Method not allowed", { status: 405 });
		}
		if (request.headers.get("upgrade") != "websocket") {
			return new Response("Endpoint is a websocket", {
				status: 426,
				headers: {
					upgrade: "websocket",
				},
			});
		}
		const { socket, response } = Deno.upgradeWebSocket(request);
		socket.addEventListener("message", (event) => {
			this.#handleSocketMessage(socket, event.data);
		});
		socket.addEventListener("close", () => {
			for (const session of this.sessions.values()) {
				session.close();
			}
			this.sessions.clear();
		});
		return response;
	}

	async #handleSocketMessage(socket, data) {
		try {
			if (typeof data != "string") {
				this.#sendError(socket, null, "Bridge expects JSON text messages.");
				return;
			}
			const message = JSON.parse(data);
			const id = message.id ?? null;
			const method = message.method;
			const payload = message.payload || {};

			if (method == "hello") {
				this.#sendSuccess(socket, id, {
					protocol_version: 2,
					tick_ms: 50,
					action_space: 5,
					capabilities: {
						step_many: true,
					},
				});
				return;
			}

			if (method == "create_env") {
				const envId = String(this.nextEnvId++);
				const parsedGlobalSeed = payload.global_seed == undefined || payload.global_seed == null
					? null
					: normalizeSeed(Number(payload.global_seed));
				const options = {
					arenaWidth: payload.arena_width || 80,
					arenaHeight: payload.arena_height || 80,
					gameMode: payload.game_mode || "default",
					opponentCount: payload.opponent_count || 7,
					maxSteps: payload.max_steps || 800,
					decisionIntervalMs: payload.decision_interval_ms || 100,
					obsRadius: payload.obs_radius || 6,
					seed: parsedGlobalSeed,
					rewardWeights: {
						score: payload.reward_score_weight ?? 0.01,
						kill: payload.reward_kill_weight ?? 1,
						death: payload.reward_death_penalty ?? -2,
						truncate: payload.reward_truncate_penalty ?? 0,
					},
				};
				const session = new BridgeSession(envId, options);
				this.sessions.set(envId, session);
				this.#sendSuccess(socket, id, {
					env_id: envId,
					options,
				});
				return;
			}

			if (method == "reset") {
				const session = this.#getSession(payload.env_id);
				const observation = session.reset();
				const resetFingerprint = `${observation.player.x}:${observation.player.y}:${observation.player.direction}`;
				this.#sendSuccess(socket, id, {
					observation,
					info: {
						env_id: payload.env_id,
						seed: session.seed,
						reset_fingerprint: resetFingerprint,
					},
				});
				return;
			}

			if (method == "observe") {
				const session = this.#getSession(payload.env_id);
				this.#sendSuccess(socket, id, {
					observation: session.observe(),
				});
				return;
			}

			if (method == "step") {
				const session = this.#getSession(payload.env_id);
				const result = await session.step(payload.action);
				this.#sendSuccess(socket, id, result);
				return;
			}

			if (method == "step_many") {
				const actions = payload.actions;
				if (!actions || typeof actions != "object" || Array.isArray(actions)) {
					this.#sendError(socket, id, "step_many payload must include an actions object mapping env_id to action.");
					return;
				}

				const entries = Object.entries(actions);
				if (entries.length == 0) {
					this.#sendError(socket, id, "step_many requires at least one env action.");
					return;
				}

				const startedAt = performance.now();
				const results = {};
				await Promise.all(entries.map(async ([envId, action]) => {
					const session = this.#getSession(envId);
					const result = await session.step(action);
					results[envId] = result;
				}));

				const elapsedMs = performance.now() - startedAt;
				this.#sendSuccess(socket, id, {
					results,
					metrics: {
						batch_size: entries.length,
						elapsed_ms: elapsedMs,
					},
				});
				return;
			}

			if (method == "close_env") {
				const session = this.#getSession(payload.env_id);
				session.close();
				this.sessions.delete(String(payload.env_id));
				this.#sendSuccess(socket, id, {
					closed: true,
				});
				return;
			}

			if (method == "ping") {
				this.#sendSuccess(socket, id, {
					pong: true,
				});
				return;
			}

			this.#sendError(socket, id, `Unknown method: ${method}`);
		} catch (err) {
			this.#sendError(socket, null, err instanceof Error ? err.message : String(err));
		}
	}

	#getSession(envId) {
		const key = String(envId || "");
		const session = this.sessions.get(key);
		if (!session) {
			throw new Error(`Unknown env_id: ${key}`);
		}
		return session;
	}

	#sendSuccess(socket, id, payload) {
		socket.send(JSON.stringify({
			id,
			ok: true,
			payload,
		}));
	}

	#sendError(socket, id, message) {
		socket.send(JSON.stringify({
			id,
			ok: false,
			error: message,
		}));
	}
}
