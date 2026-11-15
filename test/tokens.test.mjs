import assert from "node:assert/strict";
import test from "node:test";
import { estimateTokens, formatTokens } from "../lib/tokens.js";
import { toolsTokens, toolTokens } from "../lib/prompt.js";

// 宿主 @deepseek-ai/dsh-token-meter 的公式：CHARS_PER_TOKEN = 4，BLOCK_OVERHEAD = 4。
// 面板必须与它一致，否则用户核对「上下文浏览器」时永远对不上。
test("估算与宿主一致：4 字符 1 token，向上取整", () => {
  assert.equal(estimateTokens(""), 0);
  assert.equal(estimateTokens("abcd"), 1);
  assert.equal(estimateTokens("abcde"), 2);
  assert.equal(estimateTokens("x".repeat(400)), 100);
});

test("中文按字符长度算，不再 1 字 1 token", () => {
  // 17 个汉字 → ceil(17/4) = 5；旧公式会给 17，偏高 3 倍多。
  assert.equal(estimateTokens("只保留结论与文件路径，不要日志全文"), 5);
});

test("整张工具表按宿主 estimateToolsTokens 同形计算", () => {
  const tools = [
    { name: "bash", description: "Run a command", parameters: { type: "object", properties: {} } },
    { name: "edit", description: "Edit a file", parameters: { type: "object", properties: {} } },
  ];
  const expected = Math.ceil(JSON.stringify(tools).length / 4) + 4;
  assert.equal(toolsTokens(tools), expected);
  assert.equal(toolsTokens([]), 0);
});

test("逐项求和与整包估算相差很小（不加 per-item overhead）", () => {
  const tools = Array.from({ length: 20 }, (_, index) => ({
    name: `tool_${index}`,
    description: `description for tool ${index}`,
    parameters: { type: "object", properties: { a: { type: "string" } } },
  }));
  const perItem = tools.reduce((sum, tool) => sum + toolTokens(tool), 0);
  const blob = Math.ceil(JSON.stringify(tools).length / 4) + 4;
  const delta = Math.abs(perItem - blob) / blob;
  assert.ok(delta < 0.03, `逐项与整包差 ${(delta * 100).toFixed(2)}%，应小于 3%`);
});

test("formatTokens 仍是紧凑显示", () => {
  assert.equal(formatTokens(0), "0");
  assert.equal(formatTokens(999), "999");
  assert.equal(formatTokens(1234), "1.2k");
  assert.equal(formatTokens(123456), "123k");
});
