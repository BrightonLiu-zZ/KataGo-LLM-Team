#!/usr/bin/env node
/*
 * Dry-run one bot's engine the way gtp2ogs runs it, and print what would be
 * posted to the game chat.  This is the integration test for --chat: it takes
 * the *rendered* bot config, spawns `bot.command` from it, splits stderr into
 * lines exactly like gtp2ogs' Bot.ts does, and applies the chat regex read out
 * of the installed gtp2ogs bundle -- so if a gtp2ogs upgrade changes the
 * marker syntax, this fails instead of the bots going quiet on OGS.
 *
 *   node src/rank_eval/chat_probe.js config/ogs/bot-v5c.json5 [moves]
 *
 * Requires the vLLM server (src/rank_eval/serve_llm.sh start).
 */
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { execSync } = require("child_process");

const GTP2OGS_ROOT = execSync("dirname $(readlink -f $(which gtp2ogs))").toString().trim();
const NODE_MODULES = path.join(GTP2OGS_ROOT, "..", "node_modules");
const JSON5 = require(path.join(NODE_MODULES, "json5"));
const split2 = require(path.join(NODE_MODULES, "split2"));

/* The chat regex, read from the installed gtp2ogs so the two cannot drift. */
function gtp2ogsChatRegex() {
    const src = fs.readFileSync(path.join(GTP2OGS_ROOT, "gtp2ogs.js"), "utf8");
    const m = /\/\((DISCUSSION\|MALKOVICH\|MAIN)\):\(\.\*\)\//.exec(src);
    if (!m) {
        throw new Error("could not find gtp2ogs' chat regex -- did the marker syntax change?");
    }
    return new RegExp(`(${m[1]}):(.*)`);
}

async function main() {
    const config_path = process.argv[2] || "config/ogs/bot-v5c.json5";
    const n_moves = parseInt(process.argv[3] || "6", 10);
    const config = JSON5.parse(fs.readFileSync(config_path, "utf8"));
    const chat_re = gtp2ogsChatRegex();

    if (!config.bot.send_chats) {
        console.error(`FAIL: ${config_path} has bot.send_chats = false; gtp2ogs would drop every chat`);
        process.exit(1);
    }
    console.log(`config     : ${config_path}`);
    console.log(`command    : ${config.bot.command.join(" ")}`);
    console.log(`chat regex : ${chat_re}`);
    console.log(`greeting   : ${config.greeting.en}`);
    console.log(`farewell   : ${config.farewell.en}\n`);

    const proc = spawn(config.bot.command[0], config.bot.command.slice(1));
    const chats = [];
    proc.stderr.pipe(split2()).on("data", (data) => {
        const line = data.toString().trim();
        if (!line) {
            return;
        }
        const m = chat_re.exec(line);
        if (m) {
            const channel = /MALKOVICH:/i.test(line) ? "malkovich" : "main";
            chats.push({ channel, body: m[2].trim() });
            console.log(`  CHAT[${channel}] ${m[2].trim()}`);
        } else {
            console.log(`  (stderr) ${line}`);
        }
    });

    let buffer = "";
    const pending = [];
    proc.stdout.on("data", (d) => {
        buffer += d.toString();
        let i;
        while ((i = buffer.indexOf("\n\n")) !== -1) {
            const resp = buffer.slice(0, i).trim();
            buffer = buffer.slice(i + 2);
            const next = pending.shift();
            if (next) {
                next(resp);
            }
        }
    });
    const cmd = (line) =>
        new Promise((resolve) => {
            pending.push(resolve);
            proc.stdin.write(line + "\n");
        });

    await cmd("boardsize 9");
    await cmd("clear_board");
    await cmd("komi 7.5");
    let color = "b";
    for (let i = 0; i < n_moves; i++) {
        const before = chats.length;
        const resp = await cmd(`genmove ${color}`);
        const move = resp.replace(/^=\s*/, "");
        console.log(`genmove ${color} -> ${move}   (+${chats.length - before} chat)`);
        color = color === "b" ? "w" : "b";
    }
    await cmd("quit");

    const bad = chats.filter((c) => c.body.length === 0 || /[\r\n]/.test(c.body));
    console.log(`\n${chats.length} chats for ${n_moves} moves, ${bad.length} malformed`);
    const ok = chats.length === n_moves && bad.length === 0 &&
        chats.every((c) => c.channel === "main");
    console.log(ok ? "PASS: every move produced exactly one well-formed main chat"
                   : "FAIL: chat count/channel/shape is wrong");
    process.exit(ok ? 0 : 1);
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
