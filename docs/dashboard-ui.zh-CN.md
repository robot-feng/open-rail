# 运行控制台界面改进

## 项目与界面入口

Open-RAIL 把模型推理与机器人执行解耦。当前截图对应
`web_client/static/index.html`，由 `run_web_client.py` 启动的 FastAPI
服务提供，默认端口为 9000。

| 层级 | 主要入口 | 职责 |
| --- | --- | --- |
| 模型服务 | `run_server.py`、`server/models/` | 根据模型类型加载适配器，通过 ZMQ 提供推理 |
| StarVLA 适配 | `server/models/starvla/` | 模型适配、输入处理与 checkpoint 配置 |
| 机器人客户端 | `client/core/vla_client.py` | 管理观测、推理、控制、可视化及数据采集 |
| 动作处理 | `realtime_data_manager.py`、`inter_chunk_fuser.py`、`intra_chunk_smoother.py` | 缓存观测与动作，进行块间融合及块内平滑 |
| 硬件适配 | `client/robots/` | 统一机器人观测、状态、控制与动作布局 |
| Web 服务 | `web_client/server.py` | 配置、运行控制、任务指令、录制的 REST API；状态与图像数据转发 |
| 浏览器界面 | `web_client/static/` | 原生 HTML/CSS/JS 模块与 Chart.js，无前端构建依赖 |
| 离线数据查看 | `scripts/show_lerobot_data/` | 查看已经保存的 LeRobot 数据；与实时控制台入口独立 |

浏览器使用同源 `/ws` 接收运行状态，使用同源 `/visual-ws` 接收相机 JPEG
二进制帧和轨迹批次。`visual/` 还包含另一套可视化页面及共享图表库。
这次改动集中于截图中的 Web 控制台，未改变模型、SDK 或动作执行协议。

## 设计与交互

- 浅灰背景、白色卡片、蓝色主操作；绿色表示正常，琥珀色表示暂停或状态过期，红色突出停止操作。
- 普通文本使用系统无衬线字体，运行数值使用等宽字体。减少渐变、阴影和细碎边框。
- 运行指标按屏幕宽度排列；任务选择、子任务和指令编辑保持明确的阅读顺序。
- 桌面总览同时显示三路相机。保留图像比例，明确提供放大按钮、键盘入口和 Esc 返回。
- `Overview` 显示全部工作区；`Analysis` 隐藏侧栏并折叠任务正文；`Vision` 扩大相机区域，桌面上采用主相机加双腕布局。
- 视图与图例偏好保存在本机 `localStorage` 的 `openrail.workspace.v1` 中。切换这些显示选项不会调用配置保存或机器人控制接口。
- 小屏幕切换为可滚动布局，配置区位于主监控区域之后，避免控件挤压出屏幕。
- 图例默认收起，按需展开；新增缩放复位、选中关节/数据源计数与空数据引导。
- 关节选择改用原生按钮，支持 Tab、Enter/Space 与 `aria-pressed`；隐藏侧栏退出键盘焦点顺序。
- 本地 SVG 图标代替外部图标字体；支持减少动画的系统偏好。
- 轨迹窗口滑块拖动时只更新本地预览，松开后保存一次，避免连续写入配置文件。

状态条显示最近一次运行数据的时间。超过 8 秒没有新状态，显示
`UNKNOWN` 与过期提示，保留最后一次指标供参考。收到 WebSocket 或 HTTP
轮询数据后恢复；WebSocket 即使仍处于连接状态，超过时限没有数据也会启动
HTTP 备用轮询。无效状态包不会刷新数据时间。该提示反映浏览器收到的数据
是否新鲜，不代表硬件已停止，也不独立证明每路相机正在更新。

图表状态另行跟踪选中数据源的到达时间，区分暂停显示、观测停止、数据流
断连、等待数据与实时更新。图例隐藏全部曲线时会显示恢复引导。

## 查看与验证

现有 Web 服务运行时直接刷新页面即可。没有运行服务时，可按原有方式启动：

```bash
python run_web_client.py --port 9000 --conf default_conf.yaml
```

前端新交互集中在 `modules/dashboard.js`，最终主题在
`styles/dashboard.css`，图标在 `styles/icons.css`。原有业务模块与 DOM ID
继续使用，后续可以逐步将主题规则合并到对应 CSS 模块。

隔离浏览器验证：

```bash
CHROMIUM_BIN=/usr/bin/chromium-browser \
UI_SCREENSHOT_DIR=/tmp/openrail-ui-preview \
node test/dashboard_browser.cjs
```

验证使用模拟配置和遥测，不加载 checkpoint、不连接机器人。覆盖
1920、1440、1280、1151、1024、768、390、320 像素宽度下的三种视图，
以及视图恢复、侧栏展开、过期状态与 HTTP 备用轮询、相机放大退出与焦点
恢复、图例、缩放、滑块保存次数和本地资源加载。

传输测试已对齐当前同源 `/visual-ws` 协议；集成测试使用系统分配的本地端口，
验证真实可视化服务 → FastAPI 转发 → 前端处理器的 JPEG 完整性，以及文本
转发和断连清理。测试期间关闭应用 lifespan，不初始化机器人 SDK。
真实机器人视频流与长时间运行性能仍需在实际部署中确认。
