# 项目完整目录结构

保留原有相对路径；模型与重型日志需要完整 Release 包。文件逐项索引见 FILE_INDEX.md。

```text
Strong-Training/
├── docs/
│   ├── EXPERIMENT_INDEX.md
│   ├── HISTORICAL_FANHUA_README.md
│   ├── PACKAGING_POLICY.md
│   └── THIRD_PARTY.md
├── FanHua/
│   ├── configs/
│   │   ├── ablations/
│   │   │   ├── behavior_pretrain_candidate/
│   │   │   │   ├── full_seed_0.json
│   │   │   │   ├── full_seed_1.json
│   │   │   │   ├── full_seed_2.json
│   │   │   │   ├── latent_sac_seed_0.json
│   │   │   │   ├── latent_sac_seed_1.json
│   │   │   │   ├── latent_sac_seed_2.json
│   │   │   │   ├── manifest.json
│   │   │   │   ├── without_behavior_constraint_seed_0.json
│   │   │   │   ├── without_behavior_constraint_seed_1.json
│   │   │   │   ├── without_behavior_constraint_seed_2.json
│   │   │   │   ├── without_calibration_seed_0.json
│   │   │   │   ├── without_calibration_seed_1.json
│   │   │   │   ├── without_calibration_seed_2.json
│   │   │   │   ├── without_history_seed_0.json
│   │   │   │   ├── without_history_seed_1.json
│   │   │   │   └── without_history_seed_2.json
│   │   │   ├── full_seed_0.json
│   │   │   ├── full_seed_1.json
│   │   │   ├── full_seed_2.json
│   │   │   ├── latent_sac_seed_0.json
│   │   │   ├── latent_sac_seed_1.json
│   │   │   ├── latent_sac_seed_2.json
│   │   │   ├── manifest.json
│   │   │   ├── without_calibration_seed_0.json
│   │   │   ├── without_calibration_seed_1.json
│   │   │   ├── without_calibration_seed_2.json
│   │   │   ├── without_history_seed_0.json
│   │   │   ├── without_history_seed_1.json
│   │   │   └── without_history_seed_2.json
│   │   ├── prerequisites_v2/
│   │   │   ├── baseline_flappybird_seed_0.json
│   │   │   ├── baseline_flappybird_seed_1.json
│   │   │   ├── baseline_flappybird_seed_2.json
│   │   │   ├── baseline_timberman_seed_0.json
│   │   │   ├── baseline_timberman_seed_1.json
│   │   │   ├── baseline_timberman_seed_2.json
│   │   │   ├── core_full_seed_0.json
│   │   │   ├── core_full_seed_1.json
│   │   │   ├── core_full_seed_2.json
│   │   │   ├── core_without_calibration_seed_0.json
│   │   │   ├── core_without_calibration_seed_1.json
│   │   │   ├── core_without_calibration_seed_2.json
│   │   │   ├── core_without_history_seed_0.json
│   │   │   ├── core_without_history_seed_1.json
│   │   │   ├── core_without_history_seed_2.json
│   │   │   ├── focused_without_guidance_seed_0.json
│   │   │   ├── focused_without_guidance_seed_1.json
│   │   │   ├── focused_without_guidance_seed_2.json
│   │   │   ├── guided_focused_seed_0.json
│   │   │   ├── guided_focused_seed_1.json
│   │   │   ├── guided_focused_seed_2.json
│   │   │   ├── guided_focused_without_calibration_seed_0.json
│   │   │   ├── guided_focused_without_calibration_seed_1.json
│   │   │   ├── guided_focused_without_calibration_seed_2.json
│   │   │   ├── guided_focused_without_history_seed_0.json
│   │   │   ├── guided_focused_without_history_seed_1.json
│   │   │   ├── guided_focused_without_history_seed_2.json
│   │   │   ├── guided_without_focus_seed_0.json
│   │   │   ├── guided_without_focus_seed_1.json
│   │   │   ├── guided_without_focus_seed_2.json
│   │   │   ├── latent_sac_seed_0.json
│   │   │   ├── latent_sac_seed_1.json
│   │   │   └── latent_sac_seed_2.json
│   │   ├── behavior_pretrain_candidate.json
│   │   ├── formal_candidate.json
│   │   ├── mainline_shared_model.json
│   │   ├── smoke.json
│   │   └── timber_first_smoke.json
│   ├── experiments/
│   │   ├── 20261006_160242_559225_probe_timberman/
│   │   │   ├── protocol.jsonl
│   │   │   └── result.json
│   │   ├── 20261006_161425_950539_verification/
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   └── stage_01.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   └── verification.json
│   │   ├── 20261006_161444_534630_verification/
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   └── verification.json
│   │   ├── 20261006_161456_139334_timber_first_smoke/
│   │   │   ├── checkpoints/
│   │   │   │   ├── latest.pt
│   │   │   │   ├── stage_01.pt
│   │   │   │   ├── stage_02.pt
│   │   │   │   ├── stage_03.pt
│   │   │   │   └── stage_04.pt
│   │   │   ├── config.json
│   │   │   ├── episodes.csv
│   │   │   ├── protocol.jsonl
│   │   │   ├── report.json
│   │   │   ├── status.json
│   │   │   ├── training_curves.png
│   │   │   ├── updates.csv
│   │   │   └── 训练记录.md
│   │   ├── 20261006_162315_657968_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_162542_888060_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_162651_322829_evaluate_flappybird/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_162739_278874_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_162838_005406_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_163142_553244_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_163416_742012_export_timberman_training_behavior/
│   │   │   ├── dataset.json
│   │   │   └── dataset.pt
│   │   ├── 20261006_163655_182920_collect_flappybird_legacy-ppo/
│   │   │   ├── dataset.json
│   │   │   ├── dataset.pt
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_164014_242296_offline_pretrain/
│   │   │   ├── checkpoints/
│   │   │   │   └── latest.pt
│   │   │   ├── config.json
│   │   │   ├── pretrain.json
│   │   │   └── updates.csv
│   │   ├── 20261006_164133_399330_offline_pretrain/
│   │   │   ├── checkpoints/
│   │   │   │   └── latest.pt
│   │   │   ├── config.json
│   │   │   ├── pretrain.json
│   │   │   ├── report.json
│   │   │   ├── updates.csv
│   │   │   └── 训练记录.md
│   │   ├── 20261006_164202_024645_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_164310_266052_evaluate_flappybird/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_165045_902021_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   └── stage_01.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_165121_048221_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_165558_380382_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_170458_955958_collect_timberman_checkpoint/
│   │   │   ├── dataset.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_170617_192692_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_171045_665551_diagnose_timberman/
│   │   │   └── diagnosis.json
│   │   ├── 20261006_171302_768038_repair_shared_policy/
│   │   │   ├── checkpoints/
│   │   │   │   └── latest.pt
│   │   │   ├── config.json
│   │   │   ├── repair.json
│   │   │   └── repair_updates.csv
│   │   ├── 20261006_171404_213578_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_171406_403822_diagnose_timberman/
│   │   │   └── diagnosis.json
│   │   ├── 20261006_171443_253789_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_171608_803644_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_171805_279696_diagnose_timberman/
│   │   │   └── diagnosis.json
│   │   ├── 20261006_171810_268739_repair_shared_policy/
│   │   │   ├── checkpoints/
│   │   │   │   └── latest.pt
│   │   │   ├── config.json
│   │   │   ├── repair.json
│   │   │   ├── repair_updates.csv
│   │   │   ├── report.json
│   │   │   └── 训练记录.md
│   │   ├── 20261006_171857_628545_verification/
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_171910_864709_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_172118_816203_evaluate_flappybird/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_172637_611857_verification/
│   │   │   ├── guard_checkpoint/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── policy_diagnostics.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── guard_initial/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── guard_source.pt
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_173325_914610_cal_laica_guided_online/
│   │   │   ├── behavior_updates.csv
│   │   │   ├── config.json
│   │   │   ├── episodes.csv
│   │   │   ├── events.jsonl
│   │   │   ├── initial_policy_diagnostic.json
│   │   │   ├── protocol.jsonl
│   │   │   ├── status.json
│   │   │   └── updates.csv
│   │   ├── 20261006_175317_397927_verification/
│   │   │   ├── formal_gate/
│   │   │   │   ├── flappybird/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   ├── timberman/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   └── manifest.json
│   │   │   ├── guard_checkpoint/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── policy_diagnostics.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── guard_initial/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── contaminated_baseline.pt
│   │   │   ├── guard_source.pt
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261006_175422_464405_baseline_flappybird_seed_0/
│   │   │   ├── checkpoints/
│   │   │   │   ├── latest.pt
│   │   │   │   └── stage_01.pt
│   │   │   ├── behavior_updates.csv
│   │   │   ├── config.json
│   │   │   ├── episodes.csv
│   │   │   ├── events.jsonl
│   │   │   ├── protocol.jsonl
│   │   │   ├── report.json
│   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   ├── status.json
│   │   │   ├── training_curves.png
│   │   │   ├── updates.csv
│   │   │   └── 训练记录.md
│   │   ├── 20261006_193441_906918_evaluate_flappybird/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_212657_236781_baseline_timberman_seed_0/
│   │   │   ├── checkpoints/
│   │   │   │   ├── latest.pt
│   │   │   │   └── stage_01.pt
│   │   │   ├── behavior_updates.csv
│   │   │   ├── config.json
│   │   │   ├── episodes.csv
│   │   │   ├── events.jsonl
│   │   │   ├── protocol.jsonl
│   │   │   ├── report.json
│   │   │   ├── stage_01_timberman_evaluation.json
│   │   │   ├── status.json
│   │   │   ├── training_curves.png
│   │   │   ├── updates.csv
│   │   │   └── 训练记录.md
│   │   ├── 20261006_214301_209618_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_215037_511319_baseline_timberman_seed_1/
│   │   │   ├── checkpoints/
│   │   │   │   ├── latest.pt
│   │   │   │   └── stage_01.pt
│   │   │   ├── behavior_updates.csv
│   │   │   ├── config.json
│   │   │   ├── episodes.csv
│   │   │   ├── events.jsonl
│   │   │   ├── protocol.jsonl
│   │   │   ├── stage_01_timberman_evaluation.json
│   │   │   ├── status.json
│   │   │   └── updates.csv
│   │   ├── 20261006_220515_750833_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261006_221858_873810_baseline_flappybird_seed_1/
│   │   │   ├── checkpoints/
│   │   │   │   ├── latest.pt
│   │   │   │   └── stage_01.pt
│   │   │   ├── behavior_updates.csv
│   │   │   ├── config.json
│   │   │   ├── episodes.csv
│   │   │   ├── events.jsonl
│   │   │   ├── protocol.jsonl
│   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   ├── status.json
│   │   │   └── updates.csv
│   │   ├── 20261006_235401_533654_evaluate_flappybird/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261007_065645_634355_verification/
│   │   │   ├── formal_gate/
│   │   │   │   ├── flappybird/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   ├── timberman/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   └── manifest.json
│   │   │   ├── guard_checkpoint/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── policy_diagnostics.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── guard_initial/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── contaminated_baseline.pt
│   │   │   ├── guard_source.pt
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261007_065704_779897_verification/
│   │   │   ├── formal_gate/
│   │   │   │   ├── flappybird/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   ├── timberman/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   └── manifest.json
│   │   │   ├── guard_checkpoint/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── policy_diagnostics.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── guard_initial/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── contaminated_baseline.pt
│   │   │   ├── guard_source.pt
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261008_090614_484579_verification/
│   │   │   ├── formal_gate/
│   │   │   │   ├── flappybird/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   ├── timberman/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   └── manifest.json
│   │   │   ├── guard_checkpoint/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── policy_diagnostics.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── guard_initial/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── contaminated_baseline.pt
│   │   │   ├── guard_source.pt
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261008_090713_066538_evaluate_flappybird/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261008_110244_748733_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261008_113507_847365_shared_mainline_smoke/
│   │   │   ├── checkpoints/
│   │   │   │   └── initialized.pt
│   │   │   ├── config.json
│   │   │   └── preflight.json
│   │   ├── 20261008_113537_446916_shared_mainline_smoke/
│   │   │   ├── checkpoints/
│   │   │   │   ├── initialized.pt
│   │   │   │   ├── latest.pt
│   │   │   │   ├── stage_01.pt
│   │   │   │   ├── stage_02.pt
│   │   │   │   ├── stage_03.pt
│   │   │   │   └── stage_04.pt
│   │   │   ├── behavior_updates.csv
│   │   │   ├── config.json
│   │   │   ├── episodes.csv
│   │   │   ├── events.jsonl
│   │   │   ├── initial_policy_diagnostic.json
│   │   │   ├── policy_diagnostics.jsonl
│   │   │   ├── preflight.json
│   │   │   ├── protocol.jsonl
│   │   │   ├── stage_01_evaluation_cap_correction.md
│   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   ├── stage_04_timberman_evaluation.json
│   │   │   ├── status.json
│   │   │   └── updates.csv
│   │   ├── 20261008_114320_128448_verification/
│   │   │   ├── formal_gate/
│   │   │   │   ├── flappybird/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   ├── timberman/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   └── manifest.json
│   │   │   ├── guard_checkpoint/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── policy_diagnostics.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── guard_initial/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── contaminated_baseline.pt
│   │   │   ├── guard_source.pt
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261008_114339_074984_verification/
│   │   │   ├── formal_gate/
│   │   │   │   ├── flappybird/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   ├── timberman/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   └── latest.pt
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── evaluation.json
│   │   │   │   │   └── status.json
│   │   │   │   └── manifest.json
│   │   │   ├── guard_checkpoint/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── policy_diagnostics.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── guard_initial/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── initial_policy_diagnostic.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── offline_ablation/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── config.json
│   │   │   │   ├── pretrain.json
│   │   │   │   └── updates.csv
│   │   │   ├── requested_stop/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── latest.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── runner/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   ├── latest.pt
│   │   │   │   │   ├── stage_01.pt
│   │   │   │   │   ├── stage_02.pt
│   │   │   │   │   └── stage_03.pt
│   │   │   │   ├── behavior_updates.csv
│   │   │   │   ├── config.json
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── events.jsonl
│   │   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   │   ├── status.json
│   │   │   │   └── updates.csv
│   │   │   ├── contaminated_baseline.pt
│   │   │   ├── guard_source.pt
│   │   │   ├── offline_config.json
│   │   │   ├── resume_test.pt
│   │   │   ├── reused.jsonl
│   │   │   ├── socket.jsonl
│   │   │   ├── synthetic_dataset.pt
│   │   │   └── verification.json
│   │   ├── 20261008_121224_475515_evaluate_flappybird/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261008_121927_850778_evaluate_timberman/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   ├── 20261008_122246_330413_mainline_shared_model/
│   │   │   ├── checkpoints/
│   │   │   │   ├── initialized.pt
│   │   │   │   ├── latest.pt
│   │   │   │   ├── stage_01.pt
│   │   │   │   ├── stage_02.pt
│   │   │   │   ├── stage_03.pt
│   │   │   │   └── stage_04.pt
│   │   │   ├── behavior_updates.csv
│   │   │   ├── config.json
│   │   │   ├── episodes.csv
│   │   │   ├── events.jsonl
│   │   │   ├── initial_policy_diagnostic.json
│   │   │   ├── policy_diagnostics.jsonl
│   │   │   ├── preflight.json
│   │   │   ├── protocol.jsonl
│   │   │   ├── report.json
│   │   │   ├── stage_01_flappybird_evaluation.json
│   │   │   ├── stage_02_timberman_evaluation.json
│   │   │   ├── stage_03_flappybird_evaluation.json
│   │   │   ├── stage_04_timberman_evaluation.json
│   │   │   ├── status.json
│   │   │   ├── training_curves.png
│   │   │   ├── updates.csv
│   │   │   └── 训练记录.md
│   │   ├── 20261008_172006_553338_evaluate_flappybird/
│   │   │   ├── episodes.csv
│   │   │   ├── evaluation.json
│   │   │   └── protocol.jsonl
│   │   └── 20261008_200326_813205_evaluate_timberman/
│   │       ├── episodes.csv
│   │       ├── evaluation.json
│   │       └── protocol.jsonl
│   ├── fanhua/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── artifacts.py
│   │   ├── diagnostics.py
│   │   ├── envs.py
│   │   ├── evaluation.py
│   │   ├── prerequisites.py
│   │   ├── replay.py
│   │   └── supervision.py
│   ├── reference/
│   │   └── Cal-LAICA-Experiments/
│   │       ├── cal_laica/
│   │       │   ├── agent/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── Agent.py
│   │       │   │   └── SAC.py
│   │       │   ├── environments/
│   │       │   │   ├── __init__.py
│   │       │   │   └── GridWorld.py
│   │       │   ├── utils/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── Config.py
│   │       │   │   ├── Parser.py
│   │       │   │   └── utils.py
│   │       │   ├── __init__.py
│   │       │   └── runner.py
│   │       ├── data/
│   │       │   ├── ab-1m/
│   │       │   │   ├── ab2fb.txt
│   │       │   │   ├── ab_1_20.dat
│   │       │   │   ├── books.dat
│   │       │   │   └── triplet.dat
│   │       │   └── ml-1m/
│   │       │       ├── movies.dat
│   │       │       ├── ratings.dat
│   │       │       └── users.dat
│   │       ├── recsys_calql/
│   │       │   ├── agent/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── GRU.py
│   │       │   │   ├── SAC.py
│   │       │   │   └── VAE.py
│   │       │   ├── environments/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── data_loader.py
│   │       │   │   ├── funk_svd.py
│   │       │   │   └── recWorld.py
│   │       │   ├── utils/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── Config.py
│   │       │   │   ├── Parser.py
│   │       │   │   ├── Runner.py
│   │       │   │   └── utils.py
│   │       │   └── __init__.py
│   │       ├── .gitattributes
│   │       ├── .gitignore
│   │       ├── README.md
│   │       ├── requirements.txt
│   │       ├── run_cal_laica.py
│   │       └── run_recsys_calql.py
│   ├── reports/
│   │   ├── html_qa/
│   │   │   ├── calibration_paper_page5.png
│   │   │   ├── meeting_desktop.png
│   │   │   ├── meeting_method.png
│   │   │   ├── meeting_mobile.png
│   │   │   ├── meeting_results.png
│   │   │   ├── tutorial_algorithm.png
│   │   │   ├── tutorial_desktop.png
│   │   │   ├── tutorial_mobile.png
│   │   │   ├── tutorial_update.png
│   │   │   ├── verification.json
│   │   │   ├── verify_html.cjs
│   │   │   └── wemi_paper_page4.png
│   │   ├── mainline_final_evaluation.json
│   │   ├── mainline_final_flappybird_console.log
│   │   ├── mainline_final_timberman_console.log
│   │   ├── mainline_long_preflight.json
│   │   ├── mainline_preflight.json
│   │   ├── mainline_smoke_preparation.json
│   │   ├── mainline_source_checkpoint.json
│   │   ├── mainline_starting_evaluations.json
│   │   ├── runtime.json
│   │   ├── Timberman_失败定位与修复_20261006.md
│   │   ├── Timberman_重新编译复测对比_20261006.md
│   │   ├── 前置工作补齐与实验设计_v2_20261006.md
│   │   ├── 前置工作设计核对.json
│   │   ├── 前置工作设计核对_两种子.json
│   │   ├── 前置工作验收.json
│   │   ├── 单模型主线训练清单.md
│   │   ├── 单游戏主基线验收与实验设计核对_20261006.md
│   │   ├── 当前进度.json
│   │   ├── 最终总结_单模型双游戏.md
│   │   ├── 最终验收_单模型双游戏.json
│   │   ├── 知识点教程_双游戏泛化.html
│   │   ├── 算法与方法_权威论文阅读清单.md
│   │   ├── 算法迁移与验证说明.md
│   │   └── 组会汇报_双游戏泛化训练.html
│   ├── AAAI_论文中文总结_动态动作空间持续强化学习.md
│   ├── collect.py
│   ├── diagnose_timberman.py
│   ├── evaluate.py
│   ├── export_dataset.py
│   ├── finalize_collection.py
│   ├── monitor_progress.py
│   ├── prepare_configs.py
│   ├── prepare_final_evaluation.py
│   ├── prepare_mainline.py
│   ├── prepare_mainline_long.py
│   ├── prepare_mainline_smoke.py
│   ├── prepare_prerequisites.py
│   ├── pretrain.py
│   ├── probe.py
│   ├── README.md
│   ├── record_baseline.py
│   ├── repair_policy.py
│   ├── report.py
│   ├── requirements-compatible.txt
│   ├── requirements.txt
│   ├── review_prerequisites.py
│   ├── set_replication_policy.py
│   ├── train.py
│   └── validate.py
├── flappybird_rl/
│   ├── algos/
│   │   ├── __init__.py
│   │   └── ppo.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── flappybird_env.py
│   ├── figures/
│   │   ├── flappybird_episodes.png
│   │   └── flappybird_epochs.png
│   ├── logs/
│   │   ├── ppo_20260925_174115_model.pt
│   │   ├── ppo_episodes_20260925_174115.csv
│   │   ├── ppo_epochs_20260925_174115.csv
│   │   ├── smoke_20260925_173327_model.pt
│   │   ├── smoke_episodes_20260925_173327.csv
│   │   └── smoke_epochs_20260925_173327.csv
│   ├── play.py
│   ├── plot.py
│   ├── README.md
│   ├── speedtest.py
│   └── train.py
├── tools/
│   ├── publication_templates/
│   │   ├── tools/
│   │   │   ├── prepare_portable_mainline.py
│   │   │   └── verify_archive.py
│   │   ├── build_release.py
│   │   ├── gitignore.project
│   │   ├── PACKAGING_POLICY.md
│   │   ├── README.fanhua.md
│   │   ├── README.project.md
│   │   ├── requirements-compatible.txt
│   │   └── THIRD_PARTY.md
│   ├── build_release.py
│   ├── prepare_portable_mainline.py
│   └── verify_archive.py
├── WemiEnv/
│   ├── vchat_minigame/
│   │   ├── vchat_BottleJump/
│   │   │   ├── minitest/
│   │   │   │   └── test.config.json
│   │   │   ├── res/
│   │   │   │   ├── 0.png
│   │   │   │   ├── 1.png
│   │   │   │   ├── 2.png
│   │   │   │   ├── 3.png
│   │   │   │   ├── 4.png
│   │   │   │   ├── 5.png
│   │   │   │   ├── 6.png
│   │   │   │   ├── 7.png
│   │   │   │   ├── 8.png
│   │   │   │   ├── 9.png
│   │   │   │   ├── ava.png
│   │   │   │   ├── ava_b.png
│   │   │   │   ├── ava_big1.png
│   │   │   │   ├── ava_lookers.png
│   │   │   │   ├── ava_rank.png
│   │   │   │   ├── bag.png
│   │   │   │   ├── bottom.png
│   │   │   │   ├── box_bottom.png
│   │   │   │   ├── box_middle.png
│   │   │   │   ├── box_top.png
│   │   │   │   ├── btn.png
│   │   │   │   ├── btn_bg_g.png
│   │   │   │   ├── btn_bg_h.png
│   │   │   │   ├── btn_iplay.png
│   │   │   │   ├── changlle_share.png
│   │   │   │   ├── clock.png
│   │   │   │   ├── close.png
│   │   │   │   ├── combo1.mp3
│   │   │   │   ├── combo2.mp3
│   │   │   │   ├── combo3.mp3
│   │   │   │   ├── combo4.mp3
│   │   │   │   ├── combo5.mp3
│   │   │   │   ├── combo6.mp3
│   │   │   │   ├── combo7.mp3
│   │   │   │   ├── combo8.mp3
│   │   │   │   ├── cylinder_shadow.png
│   │   │   │   ├── desk_shadow.png
│   │   │   │   ├── dict.png
│   │   │   │   ├── disk.png
│   │   │   │   ├── disk_dark.png
│   │   │   │   ├── disk_light.png
│   │   │   │   ├── door.png
│   │   │   │   ├── emotion.png
│   │   │   │   ├── express.png
│   │   │   │   ├── fail.png
│   │   │   │   ├── fall.mp3
│   │   │   │   ├── fall_2.mp3
│   │   │   │   ├── flower.png
│   │   │   │   ├── flower_small.png
│   │   │   │   ├── game.png
│   │   │   │   ├── gift.png
│   │   │   │   ├── glow_bag.png
│   │   │   │   ├── gold.png
│   │   │   │   ├── golf_bottom.png
│   │   │   │   ├── golf_top.png
│   │   │   │   ├── gray.png
│   │   │   │   ├── green.png
│   │   │   │   ├── green_face.png
│   │   │   │   ├── head.png
│   │   │   │   ├── high_rank.png
│   │   │   │   ├── high_score.png
│   │   │   │   ├── high_score_week.png
│   │   │   │   ├── i.png
│   │   │   │   ├── icon.mp3
│   │   │   │   ├── indoor.png
│   │   │   │   ├── l_arr.png
│   │   │   │   ├── medicine.png
│   │   │   │   ├── money.png
│   │   │   │   ├── music_icon.png
│   │   │   │   ├── music_icon_two.png
│   │   │   │   ├── new.png
│   │   │   │   ├── noplay.png
│   │   │   │   ├── num.ttf
│   │   │   │   ├── number.png
│   │   │   │   ├── observShare.png
│   │   │   │   ├── paper_bottom.png
│   │   │   │   ├── paper_top.png
│   │   │   │   ├── papper_bottom.png
│   │   │   │   ├── perfect.mp3
│   │   │   │   ├── play.png
│   │   │   │   ├── point.png
│   │   │   │   ├── pop.mp3
│   │   │   │   ├── pure_share.png
│   │   │   │   ├── r_arr.png
│   │   │   │   ├── r_arr1.png
│   │   │   │   ├── rank.png
│   │   │   │   ├── record.png
│   │   │   │   ├── replay.png
│   │   │   │   ├── right_arr_g.png
│   │   │   │   ├── scale_intro.mp3
│   │   │   │   ├── scale_loop.mp3
│   │   │   │   ├── second_door.png
│   │   │   │   ├── shadow.png
│   │   │   │   ├── sing.mp3
│   │   │   │   ├── sing.png
│   │   │   │   ├── start.mp3
│   │   │   │   ├── stool.png
│   │   │   │   ├── stool_shadow.png
│   │   │   │   ├── store.mp3
│   │   │   │   ├── store_bottom.png
│   │   │   │   ├── store_top.png
│   │   │   │   ├── stripe.png
│   │   │   │   ├── suc.png
│   │   │   │   ├── success.mp3
│   │   │   │   ├── tired.png
│   │   │   │   ├── tit.png
│   │   │   │   ├── title.png
│   │   │   │   ├── top.png
│   │   │   │   ├── water.mp3
│   │   │   │   ├── well.png
│   │   │   │   ├── westore.png
│   │   │   │   ├── westore_desk.png
│   │   │   │   ├── white.png
│   │   │   │   └── white_face.png
│   │   │   ├── .eslintrc.js
│   │   │   ├── game.js
│   │   │   ├── game.js.map
│   │   │   ├── game.json
│   │   │   ├── interact_websocket.js
│   │   │   ├── project.config.json
│   │   │   └── README.md
│   │   ├── vchat_Flappybird/
│   │   │   ├── images/
│   │   │   │   ├── 0.gif
│   │   │   │   ├── 1.gif
│   │   │   │   ├── 2.gif
│   │   │   │   ├── bg.png
│   │   │   │   ├── ground.png
│   │   │   │   └── pipe.png
│   │   │   ├── lib/
│   │   │   │   └── wxbird.js
│   │   │   ├── pages/
│   │   │   │   ├── bird/
│   │   │   │   │   ├── bird.js
│   │   │   │   │   ├── bird.json
│   │   │   │   │   ├── bird.wxml
│   │   │   │   │   └── bird.wxss
│   │   │   │   ├── index/
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.wxml
│   │   │   │   │   └── index.wxss
│   │   │   │   └── logs/
│   │   │   │       ├── logs.js
│   │   │   │       ├── logs.json
│   │   │   │       ├── logs.wxml
│   │   │   │       └── logs.wxss
│   │   │   ├── refile/
│   │   │   │   └── WAService.js
│   │   │   ├── utils/
│   │   │   │   ├── emitter.js
│   │   │   │   ├── interact.js
│   │   │   │   └── util.js
│   │   │   ├── app.js
│   │   │   ├── app.json
│   │   │   ├── app.wxss
│   │   │   ├── project.config.json
│   │   │   └── README.md
│   │   ├── vchat_Game2048/
│   │   │   ├── images/
│   │   │   │   ├── 2048.png
│   │   │   │   └── erweima.jpg
│   │   │   ├── pages/
│   │   │   │   └── index/
│   │   │   │       ├── index.js
│   │   │   │       ├── index.wxml
│   │   │   │       └── index.wxss
│   │   │   ├── utils/
│   │   │   │   └── interact_websocket.js
│   │   │   ├── app.js
│   │   │   ├── app.json
│   │   │   ├── app.wxss
│   │   │   ├── github
│   │   │   ├── github.pub
│   │   │   ├── project.config.json
│   │   │   ├── README.md
│   │   │   └── sitemap.json
│   │   ├── vchat_GreedySnake/
│   │   │   ├── pages/
│   │   │   │   └── snake/
│   │   │   │       ├── snake.js
│   │   │   │       ├── snake.json
│   │   │   │       ├── snake.wxml
│   │   │   │       └── snake.wxss
│   │   │   ├── utils/
│   │   │   │   ├── interact_websocket.js
│   │   │   │   └── util.js
│   │   │   ├── app.js
│   │   │   ├── app.json
│   │   │   ├── app.wxss
│   │   │   ├── project.config.json
│   │   │   └── sitemap.json
│   │   ├── vchat_PlaneWar/
│   │   │   ├── .idea/
│   │   │   │   ├── inspectionProfiles/
│   │   │   │   │   └── Project_Default.xml
│   │   │   │   ├── .gitignore
│   │   │   │   ├── misc.xml
│   │   │   │   ├── modules.xml
│   │   │   │   ├── we_plain-master.iml
│   │   │   │   └── workspace.xml
│   │   │   ├── images/
│   │   │   │   ├── bg.jpg
│   │   │   │   ├── bomb.png
│   │   │   │   ├── cartridge.png
│   │   │   │   ├── cartridge_power.png
│   │   │   │   ├── die1.png
│   │   │   │   ├── die2.png
│   │   │   │   ├── loading1.png
│   │   │   │   ├── loading2.png
│   │   │   │   ├── loading3.png
│   │   │   │   ├── logo.png
│   │   │   │   ├── me.png
│   │   │   │   ├── me_2.png
│   │   │   │   ├── me_die1.png
│   │   │   │   ├── me_die2.png
│   │   │   │   ├── me_die3.png
│   │   │   │   ├── me_die4.png
│   │   │   │   ├── plain1.png
│   │   │   │   ├── plain1_die1.png
│   │   │   │   ├── plain1_die2.png
│   │   │   │   ├── plain1_die3.png
│   │   │   │   ├── plain2.png
│   │   │   │   ├── plain2_die1.png
│   │   │   │   ├── plain2_die2.png
│   │   │   │   ├── plain2_die3.png
│   │   │   │   ├── plain2_die4.png
│   │   │   │   ├── plain2_hurt.png
│   │   │   │   ├── plain3.png
│   │   │   │   ├── plain3_2.png
│   │   │   │   ├── plain3_die1.png
│   │   │   │   ├── plain3_die2.png
│   │   │   │   ├── plain3_die3.png
│   │   │   │   ├── plain3_die4.png
│   │   │   │   ├── plain3_die5.png
│   │   │   │   ├── plain3_die6.png
│   │   │   │   ├── plain3_hurt.png
│   │   │   │   ├── prop1.png
│   │   │   │   └── prop2.png
│   │   │   ├── lib/
│   │   │   │   ├── images.js
│   │   │   │   ├── wxplain.js
│   │   │   │   └── wxplain.js.back
│   │   │   ├── miniprogram_npm/
│   │   │   │   ├── base64-arraybuffer/
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   ├── css-line-break/
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   ├── eventemitter3/
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   ├── html2canvas/
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   ├── text-segmentation/
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   ├── utrie/
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   ├── widget-ui/
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   └── wxml-to-canvas/
│   │   │   │       ├── index.js
│   │   │   │       ├── index.json
│   │   │   │       ├── index.wxml
│   │   │   │       ├── index.wxss
│   │   │   │       └── utils.js
│   │   │   ├── pages/
│   │   │   │   ├── index/
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.wxml
│   │   │   │   │   └── index.wxss
│   │   │   │   ├── logs/
│   │   │   │   │   ├── logs.js
│   │   │   │   │   ├── logs.json
│   │   │   │   │   ├── logs.wxml
│   │   │   │   │   └── logs.wxss
│   │   │   │   └── plain/
│   │   │   │       ├── plain.js
│   │   │   │       ├── plain.json
│   │   │   │       ├── plain.wxml
│   │   │   │       └── plain.wxss
│   │   │   ├── utils/
│   │   │   │   ├── emitter.js
│   │   │   │   ├── interact_websocket.js
│   │   │   │   └── util.js
│   │   │   ├── app.js
│   │   │   ├── app.json
│   │   │   ├── app.wxss
│   │   │   ├── package-lock.json
│   │   │   ├── package.json
│   │   │   ├── project.config.json
│   │   │   ├── README.md
│   │   │   └── sitemap.json
│   │   ├── vchat_Timberman/
│   │   │   ├── images/
│   │   │   │   ├── bg.jpg
│   │   │   │   ├── npc.png
│   │   │   │   ├── npcDie.png
│   │   │   │   ├── npcMove.png
│   │   │   │   ├── tree.png
│   │   │   │   ├── treeBase.png
│   │   │   │   ├── treeLeft.png
│   │   │   │   └── treeRight.png
│   │   │   ├── js/
│   │   │   │   ├── base/
│   │   │   │   │   ├── bg.js
│   │   │   │   │   ├── databus.js
│   │   │   │   │   ├── gameinfo.js
│   │   │   │   │   ├── npc.js
│   │   │   │   │   ├── pool.js
│   │   │   │   │   └── tree.js
│   │   │   │   ├── libs/
│   │   │   │   │   ├── symbol.js
│   │   │   │   │   └── weapp-adapter.js
│   │   │   │   └── main.js
│   │   │   ├── demo.png
│   │   │   ├── game.js
│   │   │   ├── game.json
│   │   │   ├── project.config.json
│   │   │   └── README.md
│   │   ├── interact.js
│   │   ├── vchat_BottleJump.zip
│   │   ├── vchat_Flappybird.zip
│   │   ├── vchat_Game2048.zip
│   │   ├── vchat_GreedySnake.zip
│   │   ├── vchat_PlaneWar.zip
│   │   └── vchat_Timberman.zip
│   ├── WemiEnv/
│   │   ├── BottleJump.py
│   │   ├── Flappybird.py
│   │   ├── Game2048.py
│   │   ├── GreedySnake.py
│   │   ├── interact.py
│   │   ├── PlaneWar.py
│   │   ├── Timberman.py
│   │   └── WemiEnv.py
│   ├── .gitignore
│   └── README.md
├── 文献/
│   ├── AAAI_英文档_蒋奇奇_final_version_no_appendix.pdf
│   └── Wemi_camera_ready.pdf
├── .gitignore
└── README.md
```

目录职责和运行方式见根 README。生成目录树时，索引文件自身可能尚未落盘；最终文件全集以 ARCHIVE_MANIFEST.json 为准。
