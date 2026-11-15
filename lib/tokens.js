/**
 * Token 估算：不联网、不加载分词器。
 *
 * 口径必须和宿主一致，否则面板与「上下文浏览器」永远对不上：
 * `@deepseek-ai/dsh-token-meter/lib/types/estimate.js` 用的是固定密度
 *   CHARS_PER_TOKEN = 4，BLOCK_OVERHEAD = 4
 * 工具整包按 `ceil(JSON.stringify(tools).length / 4) + 4` 计。实测同一份
 * 45 个工具：宿主 10,793 / 逐项 ceil(len/4) 求和 10,796（差 +0.03%）。
 * 之前按「CJK 1 字 1 token、其余 3.6 字符」估，工具偏高 11.7%、段落偏高 23.3%，
 * 中文短句甚至偏高 89%，所以改成与宿主同形的纯长度公式。
 *
 * 注意：4 字符/token 对中文其实是低估（真实分词器约 1–1.5 token/汉字），
 * 但用户核对的是宿主显示的那个数，所以这里选择「跟宿主一致」而不是「更准」；
 * 界面统一标 ≈，不冒充精确值。
 */

const CHARS_PER_TOKEN = 4;

/** 估算一段文本的 token 数；空文本为 0。 */
export function estimateTokens(text) {
  if (!text) return 0;
  const value = typeof text === "string" ? text : String(text);
  return Math.ceil(value.length / CHARS_PER_TOKEN);
}

/** JSON 化后估算（工具 schema、参数）。 */
export function estimateJsonTokens(value) {
  try {
    return estimateTokens(JSON.stringify(value ?? null));
  } catch {
    return 0;
  }
}

/** 12345 → "12.3k"；用于紧凑的角标。 */
export function formatTokens(tokens) {
  const n = Math.max(0, Math.round(Number(tokens) || 0));
  if (n < 1000) return String(n);
  if (n < 100000) return `${(n / 1000).toFixed(1)}k`;
  return `${Math.round(n / 1000)}k`;
}

/** 文本首行摘要，用于列表里的一行预览。 */
export function previewOf(text, limit = 120) {
  if (!text) return "";
  const flat = String(text).replace(/\s+/g, " ").trim();
  return flat.length > limit ? `${flat.slice(0, limit)}…` : flat;
}
