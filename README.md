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
- 通过数据库里的 runtime config 动态接入多个 provider
- 当前已实现的 provider 类型：`openai-compatible`、`anthropic-native`

## 快速开始

准备环境变量：

```bash
export MOONSHOT_API_KEY=xxx
export DEEPSEEK_API_KEY=xxx
export ANTHROPIC_API_KEY=xxx
export LLMIO_ADMIN_API_KEYS=admin-secret
export LLMIO_DATABASE_PATH=./llmio.db
```

启动：

```bash
go run ./cmd/llmio
```

默认监听 `:18080`，也可以通过 `LLMIO_LISTEN` 调整。

启动后再通过管理接口写入 runtime config：

```bash
curl http://127.0.0.1:18080/admin/runtime-config \
  -X PUT \
  -H 'Authorization: Bearer admin-secret' \
  -H 'Content-Type: application/json' \
  --data @runtime-config.json.example
```

## 部署

仓库内置了一个简单发布脚本，包含：

- 本地交叉编译并打包 `tar.gz`
- 上传到远端服务器
- 安装或更新 `systemd` service
- 切换 `current` 软链并重启服务

先看帮助：

```bash
bash deploy/deploy.sh --help
```

只打包：

```bash
VERSION=v0.1.0 bash deploy/deploy.sh package
```

发布到服务器：

```bash
VERSION=v0.1.0 \
bash deploy/deploy.sh deploy
```

如果不想每次都在命令行写服务器信息，可以先准备：

```bash
cp deploy/.env.example deploy/.env
```

然后把 `deploy/.env` 改成你的实际值，脚本会自动加载它。

默认约定：

- 远端程序目录：`/opt/llmio`
- 远端配置目录：`/etc/llmio`
- systemd unit：`/etc/systemd/system/llmio.service`
- 运行用户：`llmio`

首次部署后，远端会自动生成：

- `/etc/llmio/llmio.env`
- `/etc/llmio/runtime-config.json.example`

你需要先补齐 `/etc/llmio/llmio.env` 里的环境变量，再用 `/admin/runtime-config` 把 provider、pricing、model route 写进数据库。

## 配置

启动配置改成环境变量：

```bash
LLMIO_DATABASE_PATH=/var/lib/llmio/llmio.db
LLMIO_LISTEN=:18080
LLMIO_ADMIN_API_KEYS=admin-secret
go run ./cmd/llmio
```

runtime config 通过 `GET /admin/runtime-config` 和 `PUT /admin/runtime-config` 管理，示例文件见 [runtime-config.json.example](/Users/jiayx/workspace/jiayx/le/llmio/runtime-config.json.example)。

启动环境变量：

- `LLMIO_DATABASE_PATH`：SQLite 数据库路径，API key、usage、runtime config 都存这里
- `LLMIO_LISTEN`：监听地址，默认 `:18080`
- `LLMIO_ADMIN_API_KEYS`：管理员 API key，多个值用逗号分隔

runtime config 字段说明：

- `providers[].name`：provider 名称
- `providers[].type`：后端平台类型；当前已实现 `openai-compatible`、`anthropic-native`
- `providers[].base_url`：后端 API 的 base URL，例如 `https://api.deepseek.com/v1` 或 `https://api.anthropic.com/v1`
- `providers[].api_key`：API Key，支持 `${ENV_NAME}` 环境变量展开
- `providers[].supported_api_types`：声明 provider 原生支持哪些 API 类型；为空表示默认全支持。当前已用到的值包括 OpenAI 的 `chat_completions`、`responses`，以及 Anthropic 的 `messages`
- `pricing[]`：按 `provider + backend_model` 配置计费单价，用于给 usage 汇总 `estimated_cost_usd`
- `pricing[].scheme`：计费口径；当前支持 `openai`、`anthropic`、`generic`
- `pricing[].input_per_1m_tokens`：普通输入 token 单价
- `pricing[].cached_input_per_1m_tokens`：OpenAI 风格缓存输入单价
- `pricing[].cache_read_input_per_1m_tokens`：Anthropic prompt cache read 单价
- `pricing[].cache_creation_input_per_1m_tokens`：Anthropic prompt cache write 单价
- `pricing[].output_per_1m_tokens`：输出 token 单价
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

runtime config 示例：

```json
{
  "providers": [
    {
      "name": "openai-prod",
      "type": "openai-compatible",
      "base_url": "https://api.openai.com/v1",
      "api_key": "${OPENAI_API_KEY}"
    }
  ],
  "pricing": [
    {
      "provider": "openai-prod",
      "backend_model": "gpt-5.4",
      "scheme": "openai",
      "input_per_1m_tokens": 2.5,
      "cached_input_per_1m_tokens": 0.25,
      "output_per_1m_tokens": 15
    },
    {
      "provider": "anthropic-prod",
      "backend_model": "claude-sonnet-4",
      "scheme": "anthropic",
      "input_per_1m_tokens": 3,
      "cache_read_input_per_1m_tokens": 0.3,
      "cache_creation_input_per_1m_tokens": 3.75,
      "output_per_1m_tokens": 15
    }
  ]
}
```

启用后，管理接口返回的 usage 会额外带：

- `priced_request_count`
- `estimated_cost_usd`
- `cached_input_tokens`
- `cache_read_input_tokens`
- `cache_creation_input_tokens`

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
    "name": "crm-system",
    "budget_usd": 20
  }'
```

返回里会包含一次性的 `secret`，后续业务系统拿这个 key 调用 LLM 接口。
如果设置了 `budget_usd`，超出预算后该 key 会被网关拒绝继续调用。

查看业务 API Key 和累计 token / 费用：

```bash
curl http://127.0.0.1:8080/admin/api-keys \
  -H 'Authorization: Bearer admin-secret'
```

### Runtime Config

查看当前 runtime config：

```bash
curl http://127.0.0.1:8080/admin/runtime-config \
  -H 'Authorization: Bearer admin-secret'
```

更新 runtime config：

```bash
curl http://127.0.0.1:8080/admin/runtime-config \
  -X PUT \
  -H 'Authorization: Bearer admin-secret' \
  -H 'Content-Type: application/json' \
  --data @runtime-config.json.example
```

查看所有 API Key 的 token / 费用汇总：

```bash
curl http://127.0.0.1:8080/admin/usage \
  -H 'Authorization: Bearer admin-secret'
```

查看单个 API Key 的 token / 费用：

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
