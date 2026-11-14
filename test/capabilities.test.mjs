import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  applyCaps,
  buildCatalog,
  capPlan,
  discoverDiskSkills,
  enabledToolOverrides,
  resolveCapTarget,
} from "../lib/capabilities.js";

test("没有目录段时，磁盘技能作为只读兜底列出", () => {
  const root = mkdtempSync(join(tmpdir(), "cc-skills-"));
  try {
    const dir = join(root, ".dsh", "skills", "alpha");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "SKILL.md"), "---\nname: alpha-skill\ndescription: Alpha 的磁盘摘要\n---\n# Alpha\n");
    const found = discoverDiskSkills(root).find((entry) => entry.id === "alpha-skill");
    assert.ok(found);
    assert.equal(found.readOnly, true);
    assert.equal(found.source, "filesystem");
    assert.equal(found.summary, "Alpha 的磁盘摘要");
    assert.ok(found.tokens > 0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("有目录段时以目录段为准，忽略磁盘兜底", () => {
  const diskSkills = [{
    kind: "skills",
    key: "skill:disk-only",
    id: "disk-only",
    name: "disk-only",
    summary: "磁盘",
    tokens: 0,
    readOnly: true,
  }];
  const catalog = buildCatalog({
    sections: [{ name: "toolbox:catalog", text: "- skill:real — 目录里的技能", tokens: 12 }],
    diskSkills,
  });
  assert.deepEqual(catalog.skills.map((entry) => entry.id), ["real"]);
});

test("磁盘技能也能保存三档，并生成裁剪计划", () => {
  const diskSkills = [{
    kind: "skills",
    key: "skill:disk-only",
    id: "disk-only",
    name: "disk-only",
    summary: "磁盘",
    tokens: 0,
    readOnly: true,
  }];
  const catalog = buildCatalog({ diskSkills });
  const applied = applyCaps(catalog, { "skill:disk-only": "hidden" });
  assert.equal(applied.skills[0].mode, "hidden");
  assert.equal(applied.skills[0].enabled, false);
  const plan = capPlan(catalog, { "skill:disk-only": "hidden" });
  assert.equal(plan.skills.has("disk-only"), true);
  assert.equal(plan.items.length, 0);
});

test("口语名解析成能力 key：唯一命中才返回，多义返回候选", () => {
  const entries = [
    { key: "plugin:agent-teams", id: "agent-teams", name: "agent-teams", kind: "plugins" },
    { key: "mcp:agent_team", id: "agent_team", kind: "mcp" },
    { key: "mcp:github", id: "github", kind: "mcp" },
  ];
  assert.equal(resolveCapTarget(entries, "agent-teams").entry.key, "plugin:agent-teams");
  assert.equal(resolveCapTarget(entries, "github").entry.key, "mcp:github");
  assert.equal(resolveCapTarget(entries, "nothing-like-this").kind, "none");
  assert.equal(resolveCapTarget(entries, "agent").kind, "many");
});

test("明细开关能按层级翻回能力关掉的工具，但低层 true 不能压高层 hidden", () => {
  const name = "mcp__github__issue_read";
  const rules = { tools: { [name]: { enabled: true } } };
  // 全局 hidden（第 0 层），会话显式打开（第 3 层）→ 放行
  const capLayers = [{ "mcp:github": "hidden" }, {}, {}, {}];
  const toolLayers = [{}, {}, {}, { [name]: { enabled: true } }];
  assert.deepEqual(enabledToolOverrides(rules, capLayers, toolLayers), [name]);

  // 项目 hidden（第 2 层），全局 true（第 0 层）→ 仍然关着
  const capLayers2 = [{}, {}, { "mcp:github": "hidden" }, {}];
  const toolLayers2 = [{ [name]: { enabled: true } }, {}, {}, {}];
  assert.deepEqual(enabledToolOverrides(rules, capLayers2, toolLayers2), []);
});
