#!/usr/bin/env node
// @ts-nocheck

/**
 * Extracts Splix websocket game-state packets from live web traffic.
 *
 * Usage:
 *   node scripts/extract-splix-state-playwright.js
 *
 * Optional env vars:
 *   SPLIX_URL=https://splix.io/
 *   HEADLESS=false
 *   MAX_EVENTS=200
 */

const { chromium } = require("playwright");

const SPLIX_URL = process.env.SPLIX_URL || "https://splix.io/";
const HEADLESS = String(process.env.HEADLESS || "false").toLowerCase() === "true";
const MAX_EVENTS = Number(process.env.MAX_EVENTS || 200);
const AUTO_JOIN = String(process.env.AUTO_JOIN || "false").toLowerCase() === "true";
const AUTO_NAME = process.env.AUTO_NAME || "bot-probe";

const RECEIVE_ACTION = {
  UPDATE_BLOCKS: 1,
  PLAYER_POS: 2,
  FILL_AREA: 3,
  SET_TRAIL: 4,
  PLAYER_DIE: 5,
  CHUNK_OF_BLOCKS: 6,
  REMOVE_PLAYER: 7,
  PLAYER_NAME: 8,
  MY_SCORE: 9,
  MY_RANK: 10,
  LEADERBOARD: 11,
  MAP_SIZE: 12,
  YOU_DED: 13,
  MINIMAP: 14,
  PLAYER_SKIN: 15,
  EMPTY_TRAIL_WITH_LAST_POS: 16,
  READY: 17,
  PLAYER_HIT_LINE: 18,
  REFRESH_AFTER_DIE: 19,
  PLAYER_HONK: 20,
  PONG: 21,
  UNDO_PLAYER_DIE: 22,
  TEAM_LIFE_COUNT: 23,
  PLAYER_IS_SPECTATOR: 24,
};

const ACTION_NAME = Object.fromEntries(
  Object.entries(RECEIVE_ACTION).map(([k, v]) => [v, k]),
);

function readU16BE(bytes, offset) {
  return (bytes[offset] << 8) | bytes[offset + 1];
}

function readU32BE(bytes, offset) {
  return (
    (bytes[offset] * 0x1000000) +
    (bytes[offset + 1] << 16) +
    (bytes[offset + 2] << 8) +
    bytes[offset + 3]
  ) >>> 0;
}

function readUtf8(bytes, start, end) {
  return new TextDecoder().decode(bytes.subarray(start, end));
}

function parseServerPacket(bytes) {
  if (!bytes || bytes.length < 1) return null;

  const action = bytes[0];
  const actionName = ACTION_NAME[action] || `UNKNOWN_${action}`;

  const out = {
    action,
    actionName,
    length: bytes.length,
    state: {},
  };

  switch (action) {
    case RECEIVE_ACTION.UPDATE_BLOCKS: {
      out.state = {
        x: readU16BE(bytes, 1),
        y: readU16BE(bytes, 3),
        blockType: bytes[5],
      };
      break;
    }
    case RECEIVE_ACTION.PLAYER_POS: {
      out.state = {
        x: readU16BE(bytes, 1),
        y: readU16BE(bytes, 3),
        playerId: readU16BE(bytes, 5),
        dir: bytes[7],
        extendTrail: bytes.length > 8 ? bytes[8] === 1 : undefined,
      };
      break;
    }
    case RECEIVE_ACTION.FILL_AREA: {
      out.state = {
        x: readU16BE(bytes, 1),
        y: readU16BE(bytes, 3),
        width: readU16BE(bytes, 5),
        height: readU16BE(bytes, 7),
        tileType: bytes[9],
        pattern: bytes[10],
        isEdgeChunk: bytes.length > 11 ? bytes[11] === 1 : undefined,
      };
      break;
    }
    case RECEIVE_ACTION.SET_TRAIL: {
      const playerId = readU16BE(bytes, 1);
      const vertices = [];
      for (let i = 3; i + 3 < bytes.length; i += 4) {
        vertices.push({ x: readU16BE(bytes, i), y: readU16BE(bytes, i + 2) });
      }
      out.state = { playerId, vertices, vertexCount: vertices.length };
      break;
    }
    case RECEIVE_ACTION.PLAYER_DIE: {
      out.state = {
        playerId: readU16BE(bytes, 1),
        deathPos: bytes.length > 3 ? { x: readU16BE(bytes, 3), y: readU16BE(bytes, 5) } : null,
      };
      break;
    }
    case RECEIVE_ACTION.CHUNK_OF_BLOCKS: {
      out.state = {
        x: readU16BE(bytes, 1),
        y: readU16BE(bytes, 3),
        width: readU16BE(bytes, 5),
        height: readU16BE(bytes, 7),
        blockPayloadBytes: Math.max(0, bytes.length - 9),
      };
      break;
    }
    case RECEIVE_ACTION.REMOVE_PLAYER: {
      out.state = { playerId: readU16BE(bytes, 1) };
      break;
    }
    case RECEIVE_ACTION.PLAYER_NAME: {
      out.state = {
        playerId: readU16BE(bytes, 1),
        name: readUtf8(bytes, 3, bytes.length),
      };
      break;
    }
    case RECEIVE_ACTION.MY_SCORE: {
      out.state = {
        capturedTiles: readU32BE(bytes, 1),
        kills: bytes.length > 6 ? readU16BE(bytes, 5) : 0,
      };
      break;
    }
    case RECEIVE_ACTION.MY_RANK: {
      out.state = { rank: readU16BE(bytes, 1) };
      break;
    }
    case RECEIVE_ACTION.LEADERBOARD: {
      const totalPlayers = readU16BE(bytes, 1);
      const entries = [];
      let i = 3;
      while (i + 4 < bytes.length) {
        const score = readU32BE(bytes, i);
        const nameLen = bytes[i + 4];
        const start = i + 5;
        const end = start + nameLen;
        if (end > bytes.length) break;
        entries.push({ score, name: readUtf8(bytes, start, end) });
        i = end;
      }
      out.state = { totalPlayers, entries };
      break;
    }
    case RECEIVE_ACTION.MAP_SIZE: {
      out.state = { mapSize: readU16BE(bytes, 1) };
      break;
    }
    case RECEIVE_ACTION.YOU_DED: {
      const deathType = bytes.length > 17 ? bytes[17] : undefined;
      out.state = {
        scoreTiles: bytes.length > 4 ? readU32BE(bytes, 1) : undefined,
        scoreKills: bytes.length > 6 ? readU16BE(bytes, 5) : undefined,
        highestRank: bytes.length > 8 ? readU16BE(bytes, 7) : undefined,
        timeAliveSeconds: bytes.length > 12 ? readU32BE(bytes, 9) : undefined,
        rankingFirstSeconds: bytes.length > 16 ? readU32BE(bytes, 13) : undefined,
        deathType,
        killedByName: bytes.length > 18 ? readUtf8(bytes, 18, bytes.length) : "",
      };
      break;
    }
    case RECEIVE_ACTION.MINIMAP: {
      out.state = {
        part: bytes[1],
        packedBitsLength: Math.max(0, bytes.length - 2),
      };
      break;
    }
    case RECEIVE_ACTION.PLAYER_SKIN: {
      out.state = {
        playerId: readU16BE(bytes, 1),
        skinBlock: bytes[3],
      };
      break;
    }
    case RECEIVE_ACTION.EMPTY_TRAIL_WITH_LAST_POS: {
      out.state = {
        playerId: readU16BE(bytes, 1),
        lastPos: { x: readU16BE(bytes, 3), y: readU16BE(bytes, 5) },
      };
      break;
    }
    case RECEIVE_ACTION.READY: {
      out.state = { ready: true };
      break;
    }
    case RECEIVE_ACTION.PLAYER_HIT_LINE: {
      out.state = {
        hitByPlayerId: readU16BE(bytes, 1),
        pointsColorId: bytes[3],
        x: readU16BE(bytes, 4),
        y: readU16BE(bytes, 6),
        didHitSelf: bytes.length > 8 ? bytes[8] === 1 : false,
      };
      break;
    }
    case RECEIVE_ACTION.REFRESH_AFTER_DIE: {
      out.state = { refreshAfterDie: true };
      break;
    }
    case RECEIVE_ACTION.PLAYER_HONK: {
      out.state = {
        playerId: readU16BE(bytes, 1),
        honkDuration: bytes[3],
      };
      break;
    }
    case RECEIVE_ACTION.PONG: {
      out.state = { pong: true };
      break;
    }
    case RECEIVE_ACTION.UNDO_PLAYER_DIE: {
      out.state = { playerId: readU16BE(bytes, 1), undoDie: true };
      break;
    }
    case RECEIVE_ACTION.TEAM_LIFE_COUNT: {
      out.state = {
        currentLives: bytes[1],
        totalLives: bytes[2],
      };
      break;
    }
    case RECEIVE_ACTION.PLAYER_IS_SPECTATOR: {
      out.state = { playerId: readU16BE(bytes, 1), isSpectator: true };
      break;
    }
    default: {
      out.state = { rawPreviewHex: Buffer.from(bytes.subarray(0, 24)).toString("hex") };
      break;
    }
  }

  return out;
}

function framePayloadToBytes(frame) {
  const { opcode, payloadData } = frame.response;
  if (opcode === 2) {
    return new Uint8Array(Buffer.from(payloadData, "base64"));
  }
  return new TextEncoder().encode(payloadData);
}

async function main() {
  const browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext();
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);

  await cdp.send("Network.enable");

  const seenActions = new Set();
  let events = 0;

  cdp.on("Network.webSocketCreated", (evt) => {
    console.log(`WS created: ${evt.url}`);
  });

  cdp.on("Network.webSocketFrameReceived", (evt) => {
    try {
      const bytes = framePayloadToBytes(evt);
      const parsed = parseServerPacket(bytes);
      if (!parsed) return;

      events += 1;
      seenActions.add(parsed.actionName);

      console.log("\\n--- SERVER FRAME ---");
      console.log(JSON.stringify(parsed, null, 2));

      if (events >= MAX_EVENTS) {
        console.log(`\\nReached MAX_EVENTS=${MAX_EVENTS}.`);
        console.log("Observed server game-state actions:", Array.from(seenActions).sort().join(", "));
        void context.close();
        void browser.close();
        process.exit(0);
      }
    } catch (err) {
      console.error("Failed to parse server frame:", err);
    }
  });

  await page.goto(SPLIX_URL, { waitUntil: "domcontentloaded" });

  console.log(`Opened ${SPLIX_URL}`);
  if (AUTO_JOIN) {
    try {
      await page.waitForSelector("#nameInput", { timeout: 10_000 });
      await page.fill("#nameInput", AUTO_NAME);
      await page.click("#joinButton");
      console.log(`Auto-join attempted with name: ${AUTO_NAME}`);
    } catch (err) {
      console.warn("Auto-join failed. You can join manually in headed mode.", err?.message || err);
    }
  }

  console.log("Incoming websocket state packets will be printed.");
  if (!AUTO_JOIN) {
    console.log("Interact with the page and join a game.");
  }
  console.log(`The script auto-exits after ${MAX_EVENTS} server frames.`);

  await page.waitForTimeout(24 * 60 * 60 * 1000);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
