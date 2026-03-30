# llmio

`llmio` 是一个面向多协议客户端和多平台后端的 LLM Gateway。

它对外暴露两套协议：

- OpenAI 风格接口：`/v1/chat/completions`、`/v1/responses`、`/v1/models`
- Anthropic 风格接口：`/anthropic/v1/messages`

兼容性说明：

- OpenAI 风格接口额外接受无 `/v1` 前缀的别名：`/chat/completions`、`/responses`、`/models`

当前代码已经拆成三层：

- 客户端协议适配器：把 OpenAI、Anthropic、未来的 Claude Code、Codex 等客户端请求转换成统一内部模型
- 内部标准模型：统一的 `ChatRequest / ChatResponse / StreamEvent`
- 后端 provider 适配器：把内部模型转换成具体平台协议，例如 `openai-compatible`、未来的 `anthropic-native`、`gemini-native`、`bedrock`、`vertex`

这意味着目标不是“只支持 OpenAI-compatible 后端”，而是“任意客户端协议 -> 任意后端平台”。

## 能力

- 单一网关同时暴露 OpenAI 和 Anthropic 风格 endpoint
- 模型名映射：外部模型名 -> 后端 provider target 链路
- 网关 API Key 鉴权，兼容 `Authorization: Bearer` 和 `x-api-key`
- 通过管理接口创建/停用业务 API Key
- 按业务 API Key 聚合 token 用量
- provider fallback，支持一个模型挂多个后端目标
- 客户端协议层和后端 provider 层已经解耦
- 同协议链路优先走 passthrough fast path，请求侧改写 `model`，响应侧过滤敏感 header 并重写 `model`
- 跨协议链路走统一内部模型转换
- 文本、图片块、工具定义、工具调用、工具结果都已进入统一内部表示
- OpenAI `chat/completions` 和 `responses` 都支持流式 SSE 输出，包括文本、tool call 参数增量和 reasoning 增量
- Anthropic `messages` 支持流式 SSE 输出，包括通用 `content_block_start/delta/stop` 生命周期、`tool_use` / `input_json_delta`
- 通过配置文件接入多个 provider
- 当前已实现的 provider 类型：`openai-compatible`、`anthropic-native`

## 快速开始

复制配置：

```bash
cp llmio.json.example llmio.json
```

设置密钥：

```bash
export MOONSHOT_API_KEY=xxx
export DEEPSEEK_API_KEY=xxx
export ANTHROPIC_API_KEY=xxx
export LLMIO_ADMIN_API_KEY=xxx
```

启动：

```bash
go run ./cmd/llmio
```

默认监听 `:18080`，也可以通过 `llmio.json` 中的 `listen` 调整。

## 配置

配置文件默认从 `llmio.json` 读取，也可以通过环境变量指定：

```bash
LLMIO_CONFIG=/path/to/llmio.json go run ./cmd/llmio
```

字段说明：

- `providers[].name`：provider 名称
- `providers[].type`：后端平台类型；当前已实现 `openai-compatible`、`anthropic-native`
- `providers[].base_url`：后端 API 的 base URL，例如 `https://api.deepseek.com/v1` 或 `https://api.anthropic.com/v1`
- `providers[].api_key`：API Key，支持 `${ENV_NAME}` 环境变量展开
- `providers[].supported_api_types`：声明 provider 原生支持哪些 API 类型；为空表示默认全支持。当前已用到的值包括 OpenAI 的 `chat_completions`、`responses`，以及 Anthropic 的 `messages`
- `admin_api_keys[]`：管理接口使用的管理员 API Key
- `database_path`：网关 SQLite 数据库文件路径；默认是配置文件目录下的 `llmio.db`
- `model_routes[].external_model`：对客户端暴露的模型名
- `model_routes[].targets[]`：按顺序尝试的 provider 目标列表
- `model_routes[].targets[].provider`：命中的 provider
- `model_routes[].targets[].backend_model`：实际转发给后端供应商的模型名

当后端返回 `429/500/502/503/504` 或请求直接失败时，网关会自动切到下一个 target。

passthrough 的判定规则是：

- 网关先明确知道当前入口属于哪个协议和 API 类别，例如 `openai/responses`、`openai/chat_completions`、`anthropic/messages`
- 如果命中的 provider 原生支持这个 API 类别，就走 passthrough
- 如果 provider 不支持，就回落到统一内部模型转换路径

这意味着像 Moonshot 这种只支持 `chat_completions` 的 OpenAI-compatible 后端，可以把：

```json
"supported_api_types": ["chat_completions"]
```

写进 provider 配置。这样客户端调用 `/v1/responses` 时，网关不会再去打后端的 `/responses`，而是自动转换到内部模型再通过 `/chat/completions` 落到后端。

## 架构方向

如果你的目标是让这些客户端都能接到同一个网关：

- OpenAI SDK
- Anthropic SDK
- Claude Code
- Codex

那么正确的做法不是为每个后端都复制一套 endpoint，而是：

1. 为客户端协议实现 adapter
2. 把请求归一化成内部标准模型
3. 为不同后端平台实现 provider adapter
4. 用模型路由决定外部模型名落到哪个 provider target

当前仓库已经完成第 2 步骨架，以及第 1 步中的 OpenAI / Anthropic adapter，和第 3 步中的 `openai-compatible` / `anthropic-native` provider。

后续推荐按这个顺序继续扩：

- `gemini-native` provider
- `bedrock` / `vertex` provider
- 更完整的多模态块和更细粒度的流式事件映射

## 示例

### 管理 API Key

创建业务 API Key：

```bash
curl http://127.0.0.1:8080/admin/api-keys \
  -H 'Authorization: Bearer admin-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "crm-system"
  }'
```

返回里会包含一次性的 `secret`，后续业务系统拿这个 key 调用 LLM 接口。

查看业务 API Key 和累计 token：

```bash
curl http://127.0.0.1:8080/admin/api-keys \
  -H 'Authorization: Bearer admin-secret'
```

查看所有 API Key 的 token 用量汇总：

```bash
curl http://127.0.0.1:8080/admin/usage \
  -H 'Authorization: Bearer admin-secret'
```

查看单个 API Key 的 token 用量：

```bash
curl http://127.0.0.1:8080/admin/api-keys/key_xxx/usage \
  -H 'Authorization: Bearer admin-secret'
```

### OpenAI 风格调用

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer llmio_xxx' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-proxy",
    "messages": [
      {"role": "user", "content": "你好，介绍一下你自己"}
    ]
  }'
```

### OpenAI Responses 调用

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Authorization: Bearer llmio_xxx' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-proxy",
    "input": "你好，介绍一下你自己"
  }'
```

### Anthropic 风格调用

```bash
curl http://127.0.0.1:8080/anthropic/v1/messages \
  -H 'x-api-key: llmio_xxx' \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "claude-3-5-sonnet-proxy",
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "你好，介绍一下你自己"}
    ]
  }'
```

### Anthropic 流式调用

```bash
curl http://127.0.0.1:8080/anthropic/v1/messages \
  -H 'x-api-key: llmio_xxx' \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "claude-3-5-sonnet-proxy",
    "max_tokens": 256,
    "stream": true,
    "messages": [
      {"role": "user", "content": "用三句话介绍上海"}
    ]
  }'
```

## 当前边界

- 后端 provider 目前只实现了 `openai-compatible` 和 `anthropic-native`
- fallback 目前只覆盖请求前阶段，流式响应开始输出后不会中途切换
- 同协议 passthrough 当前只做 header 和 `model` 级别的伪装，其他响应字段仍保留后端原始风格
- 非流式路径已经支持文本、图片块、工具调用和工具结果的基础映射
- 流式路径已覆盖当前仓库已实现协议面里的文本、tool call、reasoning，以及 Anthropic 通用 content block 生命周期
