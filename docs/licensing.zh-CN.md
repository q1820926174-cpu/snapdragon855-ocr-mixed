# 许可与发布范围

[返回首页](../README.zh-CN.md) · [English](licensing.md) | **简体中文**

## 公开了哪些内容

| 内容 | 许可或处理方式 |
|---|---|
| 自有集成源码、构建与验证脚本 | Apache-2.0 |
| PaddleOCR 衍生图、权重、字典和量化参数 | 保留上游 Apache-2.0 许可、署名及修改说明 |
| ncnn 补丁 | 保留上游 BSD-3-Clause 署名 |

PaddleOCR 项目及官方 [PP-OCRv6 medium 识别模型说明](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/blob/main/README.md)声明 Apache-2.0。具体署名见 [NOTICE](../NOTICE)，许可文本见 [licenses/](../licenses/)。构建依赖各自的许可仍然适用。

## 哪些高通组件需要另行获取

发布前在本地查看了已安装 SNPE 1.59 SDK 的许可。其软件、工具和文档具有独立的再分发限制；这次审查没有确认 SDK 头文件、运行库或生成的 HTA 编译缓存能够按 Apache-2.0 重新授权，因此公开版排除了这些文件。

这是本项目采取的保守发布范围，不表示所有 DLC 的单独授权分发都被禁止。

使用者需要自行合法获取 SDK、FastRPC/运行库和设备固件，并在本地生成 HTA 容器。集成代码调用 SDK API，不会改变 SDK 本身的许可。构建流程将其作为外部依赖，不下载非官方镜像，也不索取账号凭据。

## 公开包排除的文件

- 校准图、测试图、图像 raw 张量及逐图转录。
- SDK 许可 PDF、SDK 示例、头文件和第三方运行库。
- 高通或其他厂商固件、启动镜像、预编译 HTA 缓存和 ELF 执行器。
- 私人 SSH 文件、Wi-Fi 信息及私有工作区内容。

GitHub Release 发布的是单独整理的开放模型包，附有可复现的本地转换流程。最初供这台手机直接运行的私有整合包没有作为 GitHub 附件上传。
