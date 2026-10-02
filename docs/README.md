# Documentation index

This documentation describes the repository as implemented in its source files and notebooks. Start with the [repository guide](repository-guide.md) for setup and the software and hardware layout, then use the [software chapter map](software-chapters.md) to navigate each chapter. The [PPO trainer reference](ppo-trainer.md) covers the shared implementation. The [balancing bot training guide](balancing-bot-training.md) follows the complete training path from MuJoCo state through policy export.

## Audience and scope

- Learners running the notebooks and changing reward or training settings.
- Developers modifying the Gymnasium environments, PPO implementation, or inference pipeline.
- Hardware developers comparing simulator interfaces with the embedded controller.

The notebooks are executable instructional material and sometimes contain saved outputs from earlier runs. Those outputs document examples, not guaranteed results. Hardware behavior is not validated by simulator training or ONNX inference alone.
