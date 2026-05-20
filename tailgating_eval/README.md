## Tailgating Evaluation Workflow

This folder is for the current project goal: maximize tailgating detection accuracy on the 10 pilot videos.

### Subfolders

- `roi_configs/`
  One ROI config per video. Each file should store `entry`, `exit`, and `zone`.

- `manual_ground_truth/`
  Human-reviewed event labels for each video. This is the source of truth for whether a crossing was valid or tailgating.

- `system_outputs/`
  Raw event logs and summary outputs from the detector.

- `error_analysis/`
  Comparison tables and notes on why the system missed or misclassified events.

- `reports/`
  Aggregated summaries for PM/dev review.

### Intended Loop

1. Create or load ROI config for a video.
2. Run the detector on that video.
3. Save the system event log.
4. Review the video manually and save ground-truth events.
5. Compare manual ground truth vs system output.
6. Record the failure reason for every mismatch.
7. Tune code and rerun until the mismatch count drops.
